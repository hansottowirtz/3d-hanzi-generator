# 3d Hanzi Generator

## Introduction

This program generates a 3d model of a Hanzi/Kanji character using stroke data from [Make me a Hanzi](https://github.com/skishore/makemeahanzi). It extrudes the character with the stroke order as the Z dimension.

It does this by splitting up each stroke into parts following the stroke order, and then skewing each part to form a slope.

## Usage

```bash
python src/main.py --character 福 --out-scad main.scad --stl true --out-stl main.stl --settings presets/pillars_and_plate.yml
```

See [src/base_settings.yml](./src/base_settings.yml) for all configuration options.

## Installation

```bash
# Create a venv or similar, then:
pip3 install -r requirements.txt
```
