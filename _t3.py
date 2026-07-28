import os, sys, time, glob
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
SP=sys.argv[1]; M='microsoft/trocr-base-handwritten'
proc=TrOCRProcessor.from_pretrained(M); model=VisionEncoderDecoderModel.from_pretrained(M).eval()
print('TrOCR-base loaded', flush=True)
for p in sorted(glob.glob(os.path.join(SP,'vlm3','*.png'))):
    img=Image.open(p).convert('RGB')
    px=proc(images=img, return_tensors='pt').pixel_values
    t=time.time()
    with torch.no_grad(): ids=model.generate(px, max_new_tokens=24, num_beams=4)
    txt=proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    d=''.join(c for c in txt if c.isdigit())
    print(f'{os.path.basename(p):<24} -> {txt[:36]!r} digits={d!r}({len(d)}) {time.time()-t:.1f}s', flush=True)
