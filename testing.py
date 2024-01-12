import pandas as pd

import torch
import torch.backends.cudnn as cudnn

from tqdm import tqdm

from dataloading import get_dataloader
from model import load_model

            
if __name__ == '__main__':
    cudnn.benchmark = True
    
    classes = ['YOUNG', 'MIDDLE', 'OLD']
    
    model_path = 'resnet18.pt'
    batch_size = 32

    test_df = pd.read_csv('data/test.csv')
    test_df['ID'] = 'data/test/' + test_df['ID']

    test_dl = get_dataloader(test_df, batch_size=batch_size, include_labels=False, augment=False, num_workers=8)

    model = load_model(model_path)    
    model.cuda()

    pred = []
    with torch.no_grad():
        model.eval()
        for img in tqdm(test_dl):
            img= img.cuda()

            outs = model(img)

            preds = outs.argmax(dim=1)

            pred.extend([classes[i.item()] for i in preds])

    test_df['Class'] = pred
    test_df['ID'] = test_df['ID'].apply(lambda x: x.split('/')[-1])

    test_df.to_csv('submission.csv', index=False)