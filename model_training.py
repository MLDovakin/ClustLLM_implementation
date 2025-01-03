import torch
from datasets import Dataset 

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)


epochs=30
batch_size=16

train_triplets, val_triplets = train_test_split(triplets, test_size=0.2, random_state=42)

def format_triplets_for_model(triplets):
    return [InputExample(texts=[t[0], t[1], t[2]]) for t in triplets]


train_data = format_triplets_for_model(train_triplets)
val_data = format_triplets_for_model(val_triplets)

anchors = []
positive = []
negative = []
for i, triplet in enumerate(val_triplets[:2500]):
    anchors.append(df['Вопрос пользователя'].iloc[np.where(embeddings == triplet[0])[0][0]])
    positive.append(df['Вопрос пользователя'].iloc[np.where(embeddings == triplet[1])[0][0]])
    negative.append(df['Вопрос пользователя'].iloc[np.where(embeddings == triplet[2])[0][0]])


def train_model(train_data, val_data,  epochs=30, batch_size=32):

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

    dev_evaluator = TripletEvaluator(
    anchors=anchors,
    positives=positive,
    negatives=negative,
    name='embedding_model-e5',
      )
    loss = losses.TripletLoss(model=model, triplet_margin = 2)
    
    model.fit(train_objectives=[(train_dataloader, loss)],
              epochs=30, 
              optimizer_params = {'lr': 1.25e-5, 'weight_decay': 0.01, 'betas': (0.9, 0.999),},
              optimizer_class=torch.optim.AdamW, evaluation_steps = 1000, scheduler='WarmupCosine',
              warmup_steps=int(len(train_dataloader) * 0.1),  
              evaluator=dev_evaluator,
              output_path="output/sentence-transformers-model")

    return model

trained_model = train_model(train_data, val_data, epochs=30)
