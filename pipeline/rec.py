import time
from playwright.sync_api import sync_playwright
D='WORK/'
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
    ctx=b.new_context(viewport={'width':720,'height':1280},record_video_dir=D+'vid',record_video_size={'width':720,'height':1280})
    pg=ctx.new_page()
    pg.goto('file://'+D+'site/index.html')
    pg.wait_for_function('window.__ready===true')
    pg.wait_for_timeout(600)
    pg.evaluate('play()')
    END=pg.evaluate('END')
    pg.wait_for_function('window.__t>END+.3',timeout=200000) if False else pg.wait_for_timeout(int((END+1.2)*1000))
    ctx.close();b.close()
print('done',END)
