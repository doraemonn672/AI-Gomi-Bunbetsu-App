from pathlib import Path
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import WasteCNN

MODEL=Path('models/waste_cnn.pt')
CLASSES=['cardboard','glass','metal','paper','plastic','trash']
JP={'cardboard':'段ボール','glass':'ガラス','metal':'金属・缶','paper':'紙','plastic':'プラスチック','trash':'その他ごみ'}

transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
model=None

def load_model():
    global model, CLASSES
    if not MODEL.exists(): return False
    ckpt=torch.load(MODEL,map_location='cpu')
    CLASSES=ckpt.get('classes',CLASSES)
    model=WasteCNN(ckpt.get('channels',16),len(CLASSES)); model.load_state_dict(ckpt['model']); model.eval(); return True
load_model()

def predict(img):
    if img is None: return '画像を入力してください。', None
    if model is None:
        return '学習済みモデルがありません。READMEの手順で学習してください。', None
    x=transform(Image.fromarray(img).convert('RGB')).unsqueeze(0)
    with torch.no_grad(): probs=torch.softmax(model(x),1)[0]
    pairs=sorted([(JP.get(c,c),float(p)) for c,p in zip(CLASSES,probs)],key=lambda z:z[1],reverse=True)
    top=pairs[0]
    return f'判定結果：{top[0]}（{top[1]*100:.1f}%）', {k:v for k,v in pairs}

desc="写真を入力すると、CNNが6種類（段ボール・ガラス・金属・紙・プラスチック・その他ごみ）から分類します。\n※実際の分別方法は自治体のルールを確認してください。"
with gr.Blocks(title='AIゴミ分別判定アプリ') as demo:
    gr.Markdown('# ♻️ AIゴミ分別判定アプリ\n'+desc)
    with gr.Row():
        inp=gr.Image(type='numpy',label='ゴミの写真')
        out=gr.Textbox(label='CNNの判定')
    scores=gr.Label(label='予測確率',num_top_classes=6)
    btn=gr.Button('判定する',variant='primary')
    btn.click(predict,inp,[out,scores])
    gr.Markdown('### 注意\nこのアプリは画像分類の実験用です。自治体ごとに分別ルールが異なるため、最終的な判断は自治体の案内を確認してください。')

if __name__=='__main__': demo.launch()
