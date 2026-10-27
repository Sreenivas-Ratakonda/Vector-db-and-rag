"""
STEP 1 - What is an embedding?

    python rag/step1_embeddings.py

Shows three things:
  1. A sentence becomes a list of numbers (a vector).
  2. Similar meaning -> vectors point in a similar direction (high cosine similarity).
  3. Keyword search vs semantic (vector) search on the same question.
"""
from common import banner, cosine_similarity, get_embedder, keyword_score

embed = get_embedder()

# ---------------------------------------------------------------- 1. a vector
banner("1. Text -> vector")
sentence = "How do I get my money back?"
vec = embed([sentence])[0]
print(f'Sentence : "{sentence}"')
print(f"Vector   : {len(vec)} numbers")
print(f"First 8  : {[round(x, 3) for x in vec[:8]]} ...")

# ---------------------------------------------------- 2. similarity matrix
banner("2. Similar meaning = nearby vectors (cosine similarity, 1.0 = identical)")
sentences = [
    "How do I get my money back?",
    "What is the refund policy?",
    "Can I return a damaged product?",
    "How many days of annual leave do I get?",
    "My laptop was stolen, what should I do?",
]
vecs = embed(sentences)
labels = ["money back", "refund", "damaged", "leave", "laptop"]
print(" " * 14 + "".join(f"{l:>12}" for l in labels))
for label, v in zip(labels, vecs):
    row = "".join(f"{cosine_similarity(v, w):>12.2f}" for w in vecs)
    print(f"{label:<14}{row}")
print("\nRead it like a distance chart: 'money back' sits close to 'refund',")
print("far from 'leave' and 'laptop' - even with zero words in common.")

# ------------------------------------------- 3. keyword vs semantic search
banner("3. Keyword search vs vector search")
query = "I want my money back for a broken coffee machine"
corpus = [
    "Refunds go back to the original payment method within 5 to 7 business days.",
    "If a machine arrives damaged, send a photo and we reimburse the full amount.",
    "Full-time employees get 24 days of paid annual leave per year.",
    "Report a lost or stolen laptop to the IT helpdesk immediately.",
    "Keep beans in an airtight container away from sunlight.",
    "Free coffee is available in the kitchen on each floor.",
]
qv = embed([query])[0]
cv = embed(corpus)
print(f'Query: "{query}"\n')

kw = sorted(corpus, key=lambda t: -keyword_score(query, t))
sem = sorted(zip(corpus, cv), key=lambda x: -cosine_similarity(qv, x[1]))

print("Keyword search (counts shared words):")
for t in kw[:3]:
    print(f"   {keyword_score(query, t):.2f}  {t}")
print("\nVector search (compares meaning):")
for t, v in sem[:3]:
    print(f"   {cosine_similarity(qv, v):.2f}  {t}")
print("\nKeyword search latches onto 'coffee'; vector search finds 'refund'")
print("and 'damaged ... reimburse', which is what the user actually meant.")
if "fallback" in getattr(embed, "name", ""):
    print("\n[note] You are on the fallback embedder, so results look like keyword")
    print("       search. Run once with internet to download the real model.")
