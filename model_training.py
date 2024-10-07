import torch
from datasets import Dataset 

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)


train_triplets, val_triplets = train_test_split(triplets, test_size=0.2, random_state=42)

def format_triplets_for_model(triplets):
    return [InputExample(texts=[t[0], t[1], t[2]]) for t in triplets]


train_data = format_triplets_for_model(train_triplets)
val_data = format_triplets_for_model(val_triplets)

def train_model(train_data, val_data,  epochs=30, batch_size=32):

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

    dev_evaluator = TripletEvaluator(
    anchors=anchors,
    positives=positive,
    negatives=negative,
    name='embedding_model-e5',
      )
    loss = losses.TripletLoss(model=model)
    
    warmup_steps = 0
    model.fit(train_objectives=[(train_dataloader, loss)],
              epochs=epochs,
              optimizer_params = {'lr': 1e-05, 'weight_decay': 0.01},
              optimizer_class=torch.optim.RAdam,
              warmup_steps=warmup_steps,
              evaluator=dev_evaluator,  
              output_path="output/sentence-transformers-model")
    
    return model

trained_model = train_model(train_data, val_data, epochs=30)
