import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from flask import Flask, render_template, request, jsonify

export_file_url = 'https://www.googleapis.com/drive/v3/files/17bEwiC-XaRoMSYYBBhV2FdvHVa-BNKlX?alt=media&key=AIzaSyBCQLh8-VOPFr6TiN3VJQPmhd5NxXReofc'
export_file_name = 'export.pkl'

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=[
                   '*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists():
        return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


@app.route('/identify-trash', methods=['POST'])
async def identify_trash(request):
    img_data = await request.form()
    print(img_data)
    return jsonify({
        'type': 'recycle',
        'material': 'plastic',
        "confidence": 0.997
    })


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print("xwe so bad")
            return jsonify({'error': "No file found"})
    user_file = request.files['file']
    if user_file.filename == '':
        print("xwe p bad")
        return jsonify({'error': 'No file name found'})
    else:
        path = os.path.join(
            os.getcwd()+'/img/'+user_file.filename)
        user_file.save(path)

        print("YEET we good")
        # classes = identifyImage(path) TODO

        # save image details to database
        # db.addNewImage(
        #     user_file.filename,
        #     classes[0][0][1],
        #     str(classes[0][0][2]),
        #     datetime.now(),
        #     UPLOAD_URL+user_file.filename)

        return jsonify({
            'type': 'recycle',
            'material': 'plastic',
            "confidence": 0.997
        })


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
