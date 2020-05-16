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

export_file_url = 'https://drive.google.com/uc?export=download&id=17ayH-PJ39TVFm8yeMViMA_7mo2H0O9_b'
export_file_name = 'moto-trained.pkl'

classes = ['aprilia_rsv4_2009 - 12', 'aprilia_rsv4_2011 - 15', 'bmw_k-100-rs', 'bmw_k-1100-rs', 'bmw_k-75-s', 'bmw_r-1100-gs', 'bmw_r-1100-rs', 'bmw_r-1100-s', 'bmw_r-1150-gs', 'bmw_r-1200-c', 'bmw_r-1200-r_2017 - 18', 'bmw_r-nine-t_2017 - 19', 'ducati_1199-panigale_2013 - 14', 'ducati_749_2003 - 07', 'ducati_diavel-1200_2010 - 13', 'ducati_hypermotard-1100-evo_2010 - 12', 'ducati_hypermotard-1100_2007 - 09', 'ducati_hypermotard-821_2013 - 15', 'ducati_hypermotard-939_2016 - 18', 'ducati_monster-1000_2003 - 05', 'ducati_monster-1200_2014 - 16', 'ducati_monster-1200_2017 - 19', 'ducati_monster-620_2002', 'ducati_multistrada-1200_2010 - 12', 'ducati_multistrada-1200_2013 - 14', 'ducati_multistrada-1200_2015 - 17', 'ducati_multistrada-1260_2018 - 19', 'ducati_scrambler-1100_2018 - 19', 'ducati_scrambler-800_2015 - 16', 'ducati_scrambler-800_2017 - 19', 'ducati_st2_1997 - 02', 'ducati_xdiavel-1262_2016 - 19', 'harley-davidson_dyna_1994 - 99', 'harley-davidson_dyna_2007', 'harley-davidson_softail_1989 - 95', 'harley-davidson_softail_1999 - 02', 'harley-davidson_softail_1999 - 03', 'harley-davidson_softail_2006 - 07', 'harley-davidson_sportster_1994 - 00', 'harley-davidson_sportster_2001 - 05', 'harley-davidson_sportster_2006 - 07', 'harley-davidson_sportster_2008 - 12', 'harley-davidson_sportster_2013 - 17', 'harley-davidson_sportster_2014 - 16', 'harley-davidson_sportster_2018 - 19', 'harley-davidson_touring_1999 - 02', 'harley-davidson_touring_1999 - 03', 'harley-davidson_touring_2002 - 04', 'harley-davidson_touring_2003 - 05', 'harley-davidson_touring_2007', 'harley-davidson_touring_2008 - 10', 'harley-davidson_touring_2013 - 16', 'harley-davidson_touring_2014 - 16', 'harley-davidson_touring_2017', 'harley-davidson_touring_2017 - 18', 'harley-davidson_v-rod_2006 - 07', 'honda_africa-twin-crf-1000-l_2016 - 17', 'honda_cbf-1000', 'honda_crf-250-r_2018', 'honda_nc750x_2014 - 15', 'honda_nc750x_2016 -17', 'honda_nc750x_2018 - 19', 'honda_transalp-xl-700-v_2007 - 2013', 'honda_x-adv-750_2018 - 19', 'kawasaki_ninja-1000-zx-10r_2011 - 15', 'kawasaki_versys-650_2010 - 14', 'kawasaki_versys-650_2017 - 19', 'kawasaki_z-1000_2010 - 13', 'kawasaki_z-750_2007 - 14', 'ktm_1190-adventure_2013 - 16', 'ktm_exc-300-e_2018', 'kymco_people-300_2010 - 17', 'kymco_xciting-400i_2012 - 17', 'moto-guzzi_nevada-750_2002 - 06', 'moto-guzzi_stelvio-1200_2011 - 16', 'moto-guzzi_v7_2012 - 14', 'moto-guzzi_v7_2015 - 17', 'mv-agusta_brutale-1090_2012 - 15', 'mv-agusta_f4-750_2000 - 02', 'mv-agusta_turismo-veloce-800_2014 - 16', 'piaggio_mp3_2010 - 11', 'piaggio_mp3_2014 - 16', 'suzuki_v-strom-650_2008 - 11', 'triumph_speed-triple-1050_2011 - 15', 'triumph_speed-triple-955_2002 - 04', 'triumph_street-triple_2013 - 17', 'triumph_street-triple_2017 - 19', 'triumph_tiger-1050_2006 - 12', 'triumph_tiger-1200_2018 - 19', 'triumph_tiger-800_2010 - 14', 'triumph_tiger-800_2015 - 17', 'triumph_tiger-800_2018 - 19', 'triumph_tiger-explorer_2011 - 16', 'triumph_tiger-explorer_2016 - 17', 'vespa_gts-300_2014 - 16', 'yamaha_mt-09_2014 - 16', 'yamaha_t-max-530_2012 - 14', 'yamaha_t-max-530_2017 - 19', 'yamaha_xt-660_2004 - 16', 'yamaha_xt1200z-super-tenere_2015 - 16']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
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


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
