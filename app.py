import io

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from infer import VGG


model = VGG("weights/vgg.pt")
app = FastAPI()

lab2human = {-1: "empty", 0: "fake", 1: "real"}


@app.post("/process_images")
async def upload_images(files: list[UploadFile] = File(...)):
    if not files:
        return JSONResponse(content={"message": "No files provided"}, status_code=400)
    labels = []
    imgs = []
    for i, file in enumerate(files):
        try:
            contents = file.file.read()
            img = Image.open(io.BytesIO(contents))
            # label = model.forward(img)
            imgs.append(img)
            # labels.append(label)
        except Exception as e:
            return JSONResponse(
                content={"message": f"{i+1} image is not valid"}, status_code=400
            )
    labels = model.batch_forward(imgs)
    out = {
        "msg": list(map(lambda x: lab2human[x], labels)),
        "labels": labels
    }
    return JSONResponse(content=out, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
