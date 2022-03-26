# API for ChmRxnExtractor 
API Deployement of reaction synthesis extraction model.

## Installation
```bash
pip install -r requirements.txt
```

## Usage

Before starting the APIs, make sure that GROBID server is running, and ChemRxnExtraction checkpoints are placed in the correct paths defined in `.env`.

### Starting API
This project uses Flask to implement its API and can be start using the follwing command:
```bash
python main.py
```

### Creating extractions
In order to analyse the contents of a paragarph that describes a reaction, the paragraphs should be passed in a JSON object to the "/extract" endpoint of the API. For example, if the server were to be running locally in port 5000 the follwing would result in the server respoinding with the extraction to the three embedded paragraphs:

```bash
curl -X POST http://localhost:5000/extract \
-H "Content-Type: application/json" \
-d '{
        "paragraphs": [
            "First, the reaction between diphenylacetylene and tungstencarbene complex 4 was examined. At 100 C n dioxane, cycloheptadienone derivative 5 was produced in 21% yield along witha trace of rearranged cycloheptadienone 6 (Scheme 11). Longerreflux times led to formation of greater amounts of cyclo-heptadienone 6t the expense of 5. Cycloheptadienone 5 was converted to 6 after 2h at 140 C he mechanism presumablyinvolves consecutive 1,5-hydride shifts S4 In contrast to previous results obtained with alkylcarbene-tungsten complexes and alkynes, the cycloaddition reaction was never complete at 100 C. Optimal yields of cycloheptadienone 6 ( 55% ) were obtained whenthe reaction was conducted in refluxing xylene (140 C) in the presence of 1,2-bis(diphenyIphosphino) enzene.",
            "Reaction of diphenylacetylene with complex 19A led to only cycloheptadienone 23A in 30 $ yield; with (phenylcyclopropy1)- carbene complex 19B, cycloheptadienone 25 was produced in 53% yield"
        ]
    }'
```
