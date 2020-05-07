# Turbo Parser

The Turbo Parser is a dependency parser that efficiently uses higher-order
factors, such as siblings and grandparents in a dependency tree, via the 
[AD3](https://github.com/andre-martins/ad3) structured decoder. It also does 
POS (UPOS/XPOS) and morphological tagging in the 
[Universal Dependencies](https://universaldependencies.org/) style.


```
@inproceedings{turbo2020,
    title={Revisiting Higher-Order Dependency Parsers},
    author={Erick Fonseca and Andr\'{e} F. T. Martins},
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year={2020}
}
```

This code was inspired in the [original Turbo Parser](https://github.com/andre-martins/TurboParser), 
written in C++ in the pre-neural era of NLP.

## Installing

Just clone the repository and install the package with

```bash
python setup.py install
```

It should automatically install dependencies.

## Usage

### Running a trained model

For CLI usage, use the command `turboparser` (which invokes the script 
`run_parser.py` under `turboparser/scripts`). Run it with `-h` to get detailed
information. You can also check the script code for a simple API usage.


### Training a new model 

Use the command `turboparser-train` (which invokes `train_parser.py` under 
`turboparser/scripts`). 
