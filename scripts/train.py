__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

import json
import torch
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from config.hyperparameters import HYPERPARAMETERS
from config.paths import MODELS_DIR, DATASETS_DIR
from models.model import ForwardModel
from models.trainer import Trainer, initialize_model
from features.dataset import FeaturesDataset


def train(
    train_data: FeaturesDataset,
    eval_data: FeaturesDataset,
    directory: Path | str = MODELS_DIR,
    parameters: dict = HYPERPARAMETERS["forward"],
) -> ForwardModel:
    """
    Train the transformer forward model.
    :param train_data: Training dataset
    :param eval_data: Evaluation dataset
    :param directory: Directory to save trained models
    :param parameters: Training parameters
    :return: Trained model and losses
    """
    batch_size = parameters["batch_size"]
    epochs = parameters["num_epochs"]
    alpha = parameters["alpha"]
    lambda_l2 = parameters["lambda_l2"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nInitializing training...")
    
    # Setup data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size)
    
    # Initialize model
    model = initialize_model(train_data, parameters, device)
    
    # Setup optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha, weight_decay=lambda_l2)
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        parameters=parameters,
        device=device,
    )
    
    losses, best_model_state = trainer.train(epochs)

    # Save the best model and parameters to disk
    statistics = {"losses": losses}
    _save_model(model, parameters, directory, statistics, train_data, eval_data)
    return model, losses


def _save_model(
    model: ForwardModel,
    parameters: dict,
    directory: Path | str,
    statistics: dict,
    train_data: FeaturesDataset,
    eval_data: FeaturesDataset,
):
    """
    Save the trained model to disk in a timestamped directory.
    :param model: Trained ForwardModel instance
    :param parameters: Dictionary of model parameters
    :param directory: Base directory for saving the model
    :param stats: Dictionary with training statistics
    :param train_data: Training dataset
    :param eval_data: Evaluation dataset
    """
    # Timestamp directory to prevent overwriting existing models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = Path(directory) / f"forward_{timestamp}"
    directory.mkdir(parents=True, exist_ok=True)

    model_path = directory / "forward_model.pth"
    torch.save(model, model_path)

    params_path = directory / "forward_hyperparameters.json"
    with open(params_path, "w") as f:
        json.dump(parameters, f)

    manifest_path = directory / "manifest.json"
    manifest = _to_manifest(model, parameters, statistics, train_data, eval_data)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Model and parameters saved to {directory}")
    print(f"Manifest: {manifest_path}")


def _to_manifest(
    model: ForwardModel,
    parameters: dict,
    statistics: dict,
    train_data: FeaturesDataset,
    eval_data: FeaturesDataset,
) -> dict:
    """
    Create manifest dictionary with model training metadata.
    :param model: Trained ForwardModel instance
    :param parameters: Dictionary of model parameters
    :param statistics: Dictionary with training statistics
    :param train_data: Training dataset
    :param eval_data: Evaluation dataset
    :return: Manifest dictionary
    """
    return {
        "model_file": "forward_model.pth",
        "created_at": datetime.now().isoformat(),
        "train_dataset": str(train_data._path) if train_data._path else None,
        "eval_dataset": str(eval_data._path) if eval_data._path else None,
        "hyperparameters": parameters,
        "performance": statistics,
        "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


if __name__ == "__main__":
    # Find most recent dataset for training
    datasets = sorted(DATASETS_DIR.glob("dataset_*"))
    if not datasets:
        print("No datasets found. Run compile.py first.")
        exit(1)
    dataset_path = datasets[-1]
    print(f"Using dataset: {dataset_path.name}")
    
    # Find dataset files with pattern dataset_*_train.pkl
    train_files = list(dataset_path.glob("*_train.pkl"))
    eval_files = list(dataset_path.glob("*_eval.pkl"))
    if not train_files or not eval_files:
        print("Train or eval dataset not found in directory.")
        exit(1)
    
    # Load train and eval datasets
    train_data = FeaturesDataset.load(dataset_path, train_files[0].name)
    eval_data = FeaturesDataset.load(dataset_path, eval_files[0].name)
    train(train_data, eval_data, parameters=HYPERPARAMETERS["forward"])
