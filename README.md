# pix2pix
pix2pix from scratch (pytorch)


## Repository Directory 

``` python 
├── pix2pix
        ├── datasets
        │    
        ├── data
        │     ├── __init__.py
        │     └── dataset.py
        ├── option.py
        ├── model.py
        ├── train.py
        ├── inference.py
        └── README.md
```

- `data/__init__.py` : dataset loader
- `data/dataset.py` : data preprocess & get item
- `model.py` : Define block and construct Model
- `option.py` : Environment setting

<br>


## Tutoral

### Clone repo and install depenency

``` python
# Clone this repo and install dependency
git clone https://github.com/inhopp/pix2pix.git
pip install -r requirements.txt
```

<br>


### train
``` python
python3 train.py
    --device {}(defautl: cpu) \
    --lr {}(default: 0.0002) \
    --n_epoch {}(default: 200) \
    --num_workers {}(default: 4) \
    --batch_size {}(default: 4) \ 
    --eval_batch_size {}(default: 4)
```

### testset inference
``` python
python3 inference.py
    --device {}(defautl: cpu) \
    --num_workers {}(default: 4) \
    --eval_batch_size {}(default: 4)
```


<br>


#### Main Reference
https://github.com/aladdinpersson/Machine-Learning-Collection