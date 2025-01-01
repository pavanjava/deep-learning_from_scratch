import torch
import torch.nn.functional as F

dictionary = {
    "I": 1,
    "Like": 2,
    "AI": 3,
    "And": 4,
    "Machine": 5,
    "Learning": 6,
    "Data": 7,
    "Model": 8,
    "Code": 9,
    "Algorithms": 10,
    "Python": 11,
    "Research": 12,
    "Robots": 13,
    "Deep": 14,
    "Networks": 15,
    "Compute": 16,
    "Vision": 17,
    "Tools": 18,
    "Projects": 19,
    "Neural": 20,
    "Knowledge": 21,
    "Automation": 22,
    "Innovation": 23
}

sentence = "I Like Machine Learning And Deep Learning"

tensor = torch.tensor([dictionary[s] for s in sentence.split(" ")])
print(f"sentence as a tensor: {tensor}")

vocab_size = len(dictionary)  # max value is 23
embedding_dim = 4
torch.manual_seed(12)
model = torch.nn.Embedding(vocab_size, embedding_dim)
embedded_tensor = model(tensor).detach()
X = embedded_tensor
# print(embedded_tensor)
print(f"shape of embedded tensor: {embedded_tensor.shape}")
print(f"sequence length: {embedded_tensor.shape[0]} tokens")
print(f"dimension (d) of each token: {embedded_tensor.shape[1]}")

# logic of scaled dot product attention mechanism or simply self-attention mechanism
# define the hyper parameters like query, key and value dimensions d(k) d(q) and d(v)as 3
d_k = d_q = d_v = 3
d = 4

# Random weight matrices (embedding size = 4, projected size = 3)
W_q = torch.randn(d, d_q)  # For queries
W_k = torch.randn(d, d_k)  # For keys
W_v = torch.randn(d, d_v)  # For values

# Compute Q, K, V matrices
Q = torch.matmul(X, W_q)  # Shape: (7, 3)
K = torch.matmul(X, W_k)  # Shape: (7, 3)
V = torch.matmul(X, W_v)  # Shape: (7, 3)

# Print results
# Weight Matrices:
# W_q, W_k, W_v: Shape (4, 3)
# - These are the learned projection matrices that transform the input embeddings
#   from the original dimension (4) to the projected dimension (3) for queries, keys, and values.

# Projected Matrices:
# Q, K, V: Shape (7, 3)
# - Q, K, V are the resulting matrices after projecting the input embeddings.
# - 7: Number of tokens in the input sequence.
# - 3: Dimension of each token in the projected space.

print("Weight matrix W_q:\n", W_q)
print("Weight matrix W_k:\n", W_k)
print("Weight matrix W_v:\n", W_v)

print("Query matrix Q:\n", Q)
print("Key matrix K:\n", K)
print("Value matrix V:\n", V)

# Compute Attention Scores
"""
The attention scores are computed using the dot product between the query (Q) and key (K) 
vectors for each token. This gives a score that measures the similarity between each query and key.
"""
scores = torch.matmul(Q, K.transpose(-2, -1))

# Scale the Scores
"""
To prevent large values (which can destabilize training), 
scale the scores by the square root of the query/key dimension
"""
d_k = Q.size(-1)  # Query/key dimension (3 in this case)
scaled_scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

# Apply Softmax
# The scaled scores are passed through a softmax function to normalize them into probabilities (attention weights):
# Attention Weights = softmax( Scaled Scores)
attention_weights = F.softmax(scaled_scores, dim=-1)  # Shape: (7, 7)

# Compute Attention Output
# The attention output is computed by multiplying the attention weights with the value (V) matrix:
# Attention Output=Attention Weights⋅V
attention_output = torch.matmul(attention_weights, V)  # Shape: (7, 3)

# Print results
print("Scores:\n", scores)
print("Scaled Scores:\n", scaled_scores)
print("Attention Weights:\n", attention_weights)
print("Attention Output:\n", attention_output)

W_o = torch.randn(d_v, len(dictionary))  # Random weight matrix for projection
b = torch.randn(len(dictionary))  # Random bias vector

# Step 2: Compute logits for each word in the vocabulary
logits = torch.matmul(attention_output, W_o) + b  # Shape: (1, vocab_size)

# Step 3: Apply softmax to get probabilities
probabilities = F.softmax(logits, dim=-1)  # Shape: (1, vocab_size)

# Step 4: Get the predicted word (index with the highest probability)
predicted_index = torch.argmax(probabilities, dim=-1)  # Index of the predicted word

# Print results
print("Logits:\n", logits)
print("Probabilities:\n", probabilities)
print("Predicted Word Index:", predicted_index)

# Reversed Dictionary for Index-to-Word Mapping:
index_to_word = {idx: word for word, idx in dictionary.items()}
predicted_words = [index_to_word[idx] for idx in predicted_index.numpy()]

print("Predicted Words:\n", predicted_words)