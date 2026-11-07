from itertools import islice

from qdrant_client import QdrantClient, models


MODEL = "BAAI/bge-m3"


def embedder(device):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL, device=device)
    return model.get_sentence_embedding_dimension(), model.encode


def split_exact(questions):
    unique = {}
    duplicates = []
    for question in questions:
        signature = (
            question.subject, question.text, question.answer, question.choices, question.correct_key,
        )
        if signature in unique:
            duplicates.append({
                "kept_id": unique[signature].id,
                "kept_input_row": unique[signature].row,
                "removed_id": question.id,
                "input_row": question.row,
                "record": question.record,
            })
        else:
            unique[signature] = question
    return list(unique.values()), duplicates


def index_questions(client: QdrantClient, collection, questions, dimension, encode, batch_size):
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config={
                name: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
                for name in ("question", "answer")
            },
        )

    pending = iter(questions)
    while batch := list(islice(pending, batch_size)):
        stored = client.retrieve(collection, [item.id for item in batch], with_payload=False)
        stored_ids = {point.id for point in stored}
        missing = [item for item in batch if item.id not in stored_ids]
        if not missing:
            continue
        vectors = encode(
            [item.text for item in missing] + [item.answer for item in missing],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        client.upsert(collection, [
            models.PointStruct(
                id=item.id,
                vector={"question": vectors[index], "answer": vectors[index + len(missing)]},
                payload={"subject": item.subject},
            )
            for index, item in enumerate(missing)
        ], wait=True)


def find_candidates(client, collection, questions, top_k, threshold):
    by_id = {item.id: item for item in questions}
    pairs = {}
    for question in questions:
        vector = client.retrieve(
            collection, [question.id], with_payload=False, with_vectors=["answer"]
        )[0].vector["answer"]
        hits = client.query_points(
            collection,
            query=question.id,
            using="question",
            query_filter=models.Filter(
                must=[models.FieldCondition(
                    key="subject", match=models.MatchValue(value=question.subject)
                )],
                must_not=[models.HasIdCondition(has_id=[question.id])],
            ),
            limit=top_k,
            score_threshold=threshold,
            with_payload=False,
            with_vectors=["answer"],
        ).points
        for hit in hits:
            pair = tuple(sorted((question.id, hit.id)))
            if pair in pairs:
                continue
            other = by_id[hit.id]
            answer_score = sum(a * b for a, b in zip(vector, hit.vector["answer"], strict=True))
            pairs[pair] = {
                "question_ids": list(pair),
                "question_score": round(hit.score, 6),
                "answer_score": round(max(-1.0, min(1.0, answer_score)), 6),
                "same_answer_text": question.answer == other.answer,
            }
    return sorted(pairs.values(), key=lambda pair: (-pair["question_score"], pair["question_ids"]))
