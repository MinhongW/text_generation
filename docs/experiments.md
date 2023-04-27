# Experiments

Reference paper: **[Towards Table-to-Text Generation with Numerical Reasoning](https://aclanthology.org/2021.acl-long.115.pdf)**
## Naive representation + T5-small

batch_size = 4
args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1"
    #report_to="tensorboard",
)

Running time: 15:20

![image](https://user-images.githubusercontent.com/20776278/233803446-ed210fb4-573f-4521-af5a-c2d57bc3e63e.png)

The paper reports fine-tuned T5 (with naive representation) reached rouge-l 29.71. It could due to either further fine tuning or using another version of T5 model.

Tested - Freeze the first 2 layers of the encoder and the last 2 layers of the decoder

Running time: 14:30

![image](https://user-images.githubusercontent.com/20776278/233805487-7c735735-1398-46a3-9dfc-bf73cb1a3425.png)

