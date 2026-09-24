import argparse, json, random, sys
from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import WasteCNN
from data_utils import download_dataset, make_loaders


def run(channels=16, lr=1e-3, epochs=10, batch_size=32, seed=42, out="models/waste_cnn.pt"):
    random.seed(seed); torch.manual_seed(seed)
    root = download_dataset("data")
    train_loader, val_loader, test_loader, classes = make_loaders(root,batch_size,seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model=WasteCNN(channels,len(classes)).to(device)
    opt=Adam(model.parameters(),lr=lr); loss_fn=nn.CrossEntropyLoss()
    best=0
    for ep in range(epochs):
        model.train(); total=correct=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step()
            correct += (model(x).argmax(1)==y).sum().item(); total += y.numel()
        model.eval(); vc=vt=0
        with torch.no_grad():
            for x,y in val_loader:
                pred=model(x.to(device)).argmax(1); vc+=(pred==y.to(device)).sum().item(); vt+=y.numel()
        va=vc/vt
        print(f"epoch {ep+1}/{epochs}: val_accuracy={va:.4f}")
        if va>best: best=va; Path(out).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'channels':channels,'classes':classes},out)
    model.load_state_dict(torch.load(out,map_location=device)['model']); model.eval(); tc=tt=0
    with torch.no_grad():
        for x,y in test_loader:
            pred=model(x.to(device)).argmax(1); tc+=(pred==y.to(device)).sum().item(); tt+=y.numel()
    result={'channels':channels,'learning_rate':lr,'epochs':epochs,'test_accuracy':tc/tt,'classes':classes}
    Path('results').mkdir(exist_ok=True); Path('results/last_train.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--channels',type=int,default=16); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args(); run(a.channels,a.lr,a.epochs,a.batch_size)
