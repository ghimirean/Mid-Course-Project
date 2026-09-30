import wandb

# 1. Initialize W&B targeting the specific university organization
run = wandb.init(
    entity="jzg523-university-of-louisiana-monroe", 
    project="sign-language-mnist",
    name="Random-Forest-Baseline",
    notes="Week 1 Classical ML Baseline"
)

# 2. Set your Week 1 metrics 
rf_accuracy = 0.81  # Your baseline accuracy

print("Logging Random Forest baseline to W&B...")

# 3. Loop 10 times to match the 10 epochs of the Neural Network runs
for epoch in range(10):
    run.log({
        "epoch/epoch": epoch,
        "epoch/accuracy": rf_accuracy,
        "epoch/val_accuracy": rf_accuracy,
    })

run.finish()
print("Baseline successfully added to the dashboard!")