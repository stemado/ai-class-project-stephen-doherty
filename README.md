**Step-by-Step Explanation of the Method Used:**

**Objective:**  
To train a bi-encoder model capable of effectively comparing Benefit Plan names in the insurance industry by leveraging limited labeled data and augmenting it with additional data derived from our domain knowledge.

Example:

- **Benefit Plan Names:**
  - Plan A: "Voluntary Dental Premier EE"
  - Plan B: "Basic Life"


# Documentation
https://sbert.net/examples/training/data_augmentation/README.html
# What is SBERT?

SBERT (Sentence-BERT) is a modification of the BERT (Bidirectional Encoder Representations from Transformers) network that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings. These embeddings can be compared using cosine similarity, making SBERT particularly useful for tasks like semantic textual similarity, clustering, and information retrieval.

## Key Features of SBERT

- **Semantic Embeddings**: Converts sentences into dense vectors that capture their semantic meaning
- **Efficient Similarity Computation**: Allows for efficient computation of sentence similarities using cosine similarity
- **Versatile Applications**: Suitable for various NLP tasks, including semantic search, clustering, and sentence classification

## Why Use SBERT?

- **Improved Performance**: Provides better performance on sentence similarity tasks compared to traditional BERT models
- **Scalability**: Efficiently handles large datasets and can be used for real-time applications
- **Flexibility**: Can be fine-tuned on specific tasks to improve accuracy and relevance in domain-specific applications


---

### **Step 0: Data Prep**
 `training_data.tsv` and `additional_training_data.tsv`


Sample:

```tsv
train	Plus	High	1
train	Base	Basic	1
train	Basic Life	Employee Basic Life	1
train	EE	Spouse	0
```

```tsv
train   EE  EmployeeOnly    1
train   EE  Employee Only    1
train   EE  EE   1
```

### **Step 1: Import Required Libraries and Set Up Logging**

```python
import csv
import gzip
import logging
import math
import os
import shutil
import sys
from datetime import datetime

import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from collections import Counter
from sentence_transformers import LoggingHandler, SentenceTransformer, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()]
)
```

**Why:**  
We import all necessary libraries for data handling, model training, and evaluation. Setting up logging helps track the progress and debug if necessary.

---

### **Step 2: Define Parameters and Paths**

```python
# Argument Parsing
model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

batch_size = 16
num_epochs = 1
max_seq_length = 128

# Dataset Paths
sts_dataset_path = f"training/datasets/training_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.tsv.gz"
additional_sts_dataset_path = f'training/datasets/additional_training_data_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.tsv.gz'

# Output Path
bi_encoder_path = (
    "output/bi-encoder/stsb_augsbert_SS_"
    + model_name.replace("/", "-")
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
```

**Why:**  
We set default parameters for the model, define paths for datasets and outputs, and allow for command-line overrides. This ensures flexibility and organization in managing different runs and datasets.

---

### **Step 3: Load the Initial Bi-Encoder Model**

```python
# Load Bi-Encoder
logging.info(f"Loading bi-encoder model: {model_name}")
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)
bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

**Why:**  
We initialize the bi-encoder model with a pre-trained transformer (e.g., BERT). This model generates sentence embeddings which we will fine-tune on our specific task.

---

### **Step 4: Compress and Verify Training Data Files**

```python
def compress_file(input_path, output_path, buffer_size=1024 * 1024):
    # Function to compress files

def verify_compression(original, compressed, temp_path):
    # Function to verify compressed files

# Compress the files
compress_file('datasets/training_data.tsv', sts_dataset_path)
compress_file('datasets/additional_training_data.tsv', additional_sts_dataset_path)

# Verify the compression
verify_compression('datasets/training_data.tsv', sts_dataset_path, 'datasets/temp_decompressed.tsv')
```

**Why:**  
Compressing data saves storage space and speeds up data loading. Verifying ensures data integrity after compression.

---

### **Step 5: Load and Process the Dataset**

```python
gold_samples = []

logging.info("Loading and processing dataset...")
with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row_number, row in enumerate(reader, start=1):
        try:
            original_score = int(float(row["score"]))
            label = 1 if original_score == 1 else 0
            if row["split"] == "train":
                # Add both (A, B) and (B, A) for symmetry
                gold_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=label))
                gold_samples.append(InputExample(texts=[row["sentence2"], row["sentence1"]], label=label))
        except Exception as e:
            print(f"Error processing row {row_number}: {row}")
            print(f"Exception: {e}")
```

**Why:**  
We load the training data, converting scores to binary labels (similar or not similar). Adding both (A, B) and (B, A) pairs ensures that the model learns that sentence order doesn't affect similarity.

---

### **Step 6: Split Data into Training, Validation, and Test Sets**

```python
labels = [example.label for example in gold_samples]

# Split into training and temporary sets
train_samples, temp_samples = train_test_split(
    gold_samples,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=labels
)

# Split temporary set into validation and test sets
temp_labels = [example.label for example in temp_samples]
dev_samples, test_samples = train_test_split(
    temp_samples,
    test_size=0.5,
    random_state=42,
    shuffle=True,
    stratify=temp_labels
)
```

**Why:**  
We split the data while maintaining the class distribution (stratification). This ensures that each set (training, validation, test) is representative of the overall data, which is crucial for reliable model evaluation.

---

### **Step 7: Generate the Silver Dataset Using Semantic Search**

```python
# Step 2.1: Generate silver dataset
silver_data = []
sentences = set()

for sample in gold_samples:
    sentences.update(sample.texts)

sentences = list(sentences)
sent2idx = {sentence: idx for idx, sentence in enumerate(sentences)}
duplicates = set((sent2idx[data.texts[0]], sent2idx[data.texts[1]]) for data in train_samples)

# Encode sentences using a pre-trained model
semantic_model_name = "paraphrase-MiniLM-L6-v2"
semantic_search_model = SentenceTransformer(semantic_model_name)
embeddings = semantic_search_model.encode(sentences, batch_size=batch_size, convert_to_tensor=True)

# Retrieve top-k similar sentences
for idx in range(len(sentences)):
    sentence_embedding = embeddings[idx]
    cos_scores = util.cos_sim(sentence_embedding, embeddings)[0]
    cos_scores = cos_scores.cpu()

    top_results = torch.topk(cos_scores, k=top_k + 1)

    for score, iid in zip(top_results[0], top_results[1]):
        if iid != idx and (iid, idx) not in duplicates:
            silver_data.append((sentences[idx], sentences[iid]))
            duplicates.add((idx, iid))
```

**Why:**  
To augment our dataset, we generate additional sentence pairs (silver data) by finding top-k similar sentences using semantic search. This leverages the unlabeled data to create meaningful pairs that can enhance the model's learning.

---

### **Step 8: Label the Silver Dataset Using the Bi-Encoder**

```python
# Step 2.2: Label silver dataset
silver_scores = bi_encoder.predict(silver_data)

# Create InputExamples for silver dataset
silver_samples = [
    InputExample(texts=[pair[0], pair[1]], label=score)
    for pair, score in zip(silver_data, silver_scores)
]
```

**Why:**  
We use the bi-encoder to assign similarity scores to the silver data pairs. This effectively labels the new data, allowing us to include it in the training process.

---

### **Step 9: Combine Gold and Silver Data for Training**

```python
combined_train_samples = train_samples + silver_samples
combined_train_dataloader = DataLoader(combined_train_samples, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model=bi_encoder)
```

**Why:**  
Combining both datasets provides more training data, which can improve the model's ability to generalize and understand nuances in Benefit Plan names.

---

### **Step 10: Train the Bi-Encoder Model**

```python
bi_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name="sts-dev")

warmup_steps = math.ceil(len(combined_train_dataloader) * num_epochs * 0.1)
logging.info(f"Warmup-steps: {warmup_steps}")

bi_encoder.fit(
    train_objectives=[(combined_train_dataloader, train_loss)],
    evaluator=bi_evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=bi_encoder_path,
    use_amp=True,
)
```

**Why:**  
We train the bi-encoder on the combined dataset using cosine similarity loss, which is appropriate for measuring sentence similarity. The evaluator monitors performance on the validation set to prevent overfitting.

---

### **Step 11: Evaluate the Final Model**

```python
bi_encoder = SentenceTransformer(bi_encoder_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name="sts-test")
test_evaluator(bi_encoder, output_path=bi_encoder_path)
```

**Why:**  
Evaluating on the test set gives an unbiased assessment of the model's performance on unseen data, ensuring that it generalizes well.

---

### **Summary of the Method and Rationale:**

1. **Data Preparation and Symmetry:**  
   - **Method:** Loaded and processed the gold dataset, converting scores to binary labels and adding symmetric pairs.
   - **Why:** Symmetric pairs (A, B) and (B, A) help the model understand that the order of sentences doesn't affect similarity.

2. **Data Splitting with Stratification:**  
   - **Method:** Split data into training, validation, and test sets while maintaining class distribution.
   - **Why:** Ensures that each set is representative, which is crucial for reliable evaluation.

3. **Data Augmentation (Silver Data Generation):**  
   - **Method:** Generated additional sentence pairs using semantic search and a pre-trained model.
   - **Why:** Augments limited labeled data with meaningful pairs, improving the model's learning capacity.

4. **Labeling Silver Data with Bi-Encoder Predictions:**  
   - **Method:** Used the bi-encoder to assign similarity scores to the silver data.
   - **Why:** Provides approximate labels for new data, enabling it to contribute to training.

5. **Combined Training:**  
   - **Method:** Trained the bi-encoder on both gold and silver data using cosine similarity loss.
   - **Why:** Combines high-quality labels with augmented data to improve generalization.

6. **Evaluation:**  
   - **Method:** Evaluated the model on the validation set during training and on the test set after training.
   - **Why:** Monitors performance to prevent overfitting and assess generalization.

---

### **Benefits for Comparing Benefit Plan Names:**

- **Domain Adaptation:**  
  Training on domain-specific data (Benefit Plan names) allows the model to understand industry-specific terminology and nuances.

- **Improved Similarity Assessment:**  
  Augmented data helps the model better capture variations and similarities between different plan names, leading to more accurate comparisons.

- **Efficient Use of Limited Data:**  
  By generating and labeling additional data, we overcome the challenge of limited labeled examples in specialized domains.

- **Model Generalization:**  
  Combining gold and silver data enhances the model's ability to generalize from known examples to new, unseen plan names.

---

**Conclusion:**

By following this method, we effectively trained a bi-encoder model tailored to compare Benefit Plan names in the insurance industry, leveraging both our existing domain knowledge and augmented data to overcome data scarcity and improve performance.

