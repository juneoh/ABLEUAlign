# ABLEUAlign

An implementation of [ABLEUAlign](https://brunch.co.kr/@kakao-it/154) using
pretrained word embeddings from [torchtext](https://github.com/pytorch/text).

## Installation

```bash
pip install git+https://github.com/juneoh/ableualign
```

## Usage

**Please note that the pretrained vocabulary of over 2GB will be downloaded upon
first run.**

```bash
ableualign -t target.txt -r reference.txt -o output.txt -p
```

See `ableualign --help` for details.
