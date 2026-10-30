from fastapi import FastAPI, UploadFile, File, HTTPException
from backend.factory import LoaderFactory
from backend.data_manager import DataManager
from io import BytesIO

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "Healthy"}
    
@app.post("/dataset/summary")
async def dataset_summary(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        source = BytesIO(contents)
        source.name = file.filename
        loader = LoaderFactory.create_loader(source)
        manager = DataManager(loader)
        manager.load()

        row, column = manager.analyzer.shape()
        column_names = manager.analyzer.columns().to_list()
        missing_values = int(manager.analyzer.missing_values().sum())
        duplicate_rows = len(manager.analyzer.duplicates())
        return {
            "filename": file.filename,
            "rows": row,
            "columns": column,
            "column_names": column_names,
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows
        }
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error