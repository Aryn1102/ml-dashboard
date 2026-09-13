from backend.factory import LoaderFactory
from backend.data_manager import DataManager
loader = LoaderFactory.create_loader("train.csv")

manager = DataManager(loader)

manager.load()

print(manager.analyzer.shape())

print(manager.analyzer.missing_values())

manager.cleaner.normalize_missing_values()

manager.cleaner.fill_missing({"age":30})

manager.cleaner.standardize_text({"sex",
    "class",
    "deck",
    "embark_town"})

manager.cleaner.remove_duplicates()

print(manager.analyzer.shape())

print(manager.analyzer.missing_values())

print(manager.analyzer.duplicates())

print(manager.analyzer.unique_values("sex"))

manager.save("cleaned.csv")