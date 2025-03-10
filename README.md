[![AI4EF-Logo](./docs/figures/color%20all.png)
](https://raw.githubusercontent.com/epu-ntua/enershare-ai4ef/main/docs/figures/color%20logo.png)

#
[![DOI](https://zenodo.org/badge/733031403.svg)](https://doi.org/10.5281/zenodo.12205373)
[![YouTube](https://img.shields.io/badge/YouTube-Watch%20Now-red)](https://www.youtube.com/watch?v=JDTy8DvhDvA)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

The repository for AI4F services for updating building energy class and forecasting effect of potential benefit at installing solar panels.

## Installation

This project is implemented in [Docker]((https://docs.docker.com/)) providing a complete image for the entire service -- see **Dockerfile** and **docker_compose.yml** file for details regarding installation and requirements/dependencies

| Entrypoint                 |            Filename           |
|:--------------------------:|:-----------------------------:|
| api                        | api.py                        |
| Investment Planning        | MLPClassifier_HPO.py          |
| Photovoltaic Installation  | MLPRegresssor_HPO.py          |

The project also includes:
* **Dockerfile** and **docker_compose.yml**: docker files responsible for deploying the respective image, as well as **python_requirements.txt** that contains the pip dependencies required to do so.
* **service1_outputs.json** and **service1_outputs.json**: JSON files that contain each product target’s name, description, data type. Paired with the respective model forecasts, they are utilized by API as format for the user response sent to front-end.

```bash
docker compose build
docker compose up -d
```
After installation is complete, you should see the official AI4EF dashboard located at `localhost:3000`, the AI4EF MLApp must be visible at `localhost:8888` and the AI4EF Training Playground app UI must be visible in `localhost:8890`.
The code for the frontend is provided on a seperate repo at: [https://github.com/epu-ntua/enershare-ai4ef-frontend](https://github.com/epu-ntua/enershare-ai4ef-frontend) 

## Data format

Our pipelines are currently capable of processing a single csv file at a time, for the purposes of the service. 
Data must: 
* be in csv format 
* contain the columns registered either by the user-provided values or default values of the commnand-line arguments (see service 2 for more details)
  
Please fill command line arguments refering to columns of the dataset provided, otherwise the code will **not** function properly
Services use python's [click](https://click.palletsprojects.com/en/8.1.x/api/) to edit parameters using command line arguments. Please refer to each service's documentation for further details.

<!--

### Service 1: Invesment Planning (classifier)

Service 1 classification provides cross-validation grid search among many models and hyperparameter values to determine the most suitable combination:

| Algorithm      | Parameters                                                                                                        |
|----------------|-------------------------------------------------------------------------------------------------------------------|
| KNN            | {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15], 'weights': ['uniform', 'distance']}                                |
| SVC            | {'estimator__kernel': ['rbf', 'linear'], 'estimator__gamma': ['scale', 'auto'], 'estimator__C': [10, 100, 1000]}  |
| LR             | {'estimator__penalty': ['none', 'l2'], 'estimator__C': [1.0, 0.5, 0.1], 'estimator__solver': ['lbfgs', 'liblinear']}|
| Decision Tree  | {'criterion': ['entropy'], 'max_depth': [6], 'min_samples_leaf': [1], 'min_samples_split': [4]}                  |
| Random Forest  | {'n_estimators': [200, 600], 'max_depth': [4, 10, None], 'min_samples_leaf': [1, 2, 5]}                           |
| MLP            | {'hidden_layer_sizes': [150, 100, 50], 'activation': ['relu', 'logistic', 'tanh'], 'solver': ['adam', 'lbfgs', 'sgd']}|
| XGBoost        | {'learning_rate': [0.1, 0.2, 0.3], 'max_depth': [1, 2, 3, 4, 5, 6], 'min_child_weight': [1, 2], 'subsample': [1.0, 0.5, 0.1], 'n_estimators': [200, 600]}|

#### Command-line arguments

|   Parameters   | Type |        Default Value       |                           Description                           |
|:--------------:|:----:|:--------------------------:|:---------------------------------------------------------------:|
| input_filepath |  str |    './EF_comp.csv'         |        Folder path containing csv files used by the model       |
|   feature_cols |  str |              -             |       Dataset columns necesary for training                     |
|   target_cols  |  str |              -             |       Target column that we want to predict (model output)      |
|   output_dir   |  str |     './models-scalers/'    |            local directory path to store models/scalers         |

**Example:** 
```bash
python classifier.py --input_filepath ./EF_comp.csv --feature_cols Building total area,Reference area,Above ground floors,Underground floor,Initial energy class,Energy consumption before,Energy class after
                     --target_cols Carrying out construction works,Reconstruction of engineering systems,Heat installation,Water heating system
                     --output_dir ./models-scalers/ 
```
-->

## MLApp (Investment Planning / Photovolatic Installation)

### FastAPI 

FastAPI servers are the intermediate between the user interface and our model services, handling user requests, generating forecasts and providing the appropriate responses
it uses swagger UI that, upon deployment, generates documentation that can be foun in localhost with port 8888 [](http://enershare.epu.ntua.gr:8888/docs#/) 
It deploys two (2) endpoints, one for each service with the following openAPI descriptions

#### Investment Planning Endpoint

| Description             | Get building parameters service 1                              |
|-------------------------|----------------------------------------------------------------|
| HTTP Method             | POST                                                           |
| Endpoint URL            | `<host_ip>::8888/service_1/inference`                          |
| Parameters              | [{"title":"Building Total Area","description":"Total area of the building. It is determined by summing up the entire room, including the basement floor, plinth floor, technical floor, attic floor, if the height of the relevant part of the interior space from the floor to the lower surface of the structure is at least 1.6 meters","type":"float","unit":"m^2","minimum":0,"maximum":null,"id":"1"},{"title":"Above-ground floors","description":"Number of above-ground floors","type":"int","unit":"-","minimum":1,"maximum":3,"id":"3"},{"title":"Initial energy class","description":"The initial energy efficiency class of the house ranges from A+ to G class","type":"object","unit":"-","minimum":"G","maximum":"A+","id":"5"},{"title":"Energy consumption before","description":"Inverter set power in project - in addition to the existing inverter.","type":"float","unit":"kWh/m^2","minimum":0,"maximum":null,"id":"6"},{"title":"Energy class after","description":"Energy efficiency class according to energy audit (after the renovation of the building). Energy efficiency class of the house ranges from A+ to F class","type":"object","unit":"-","minimum":"G","maximum":"A+","id":"7"}]   |
| Output example          | [{"title":"Carrying out construction works","description":"Carrying out construction works in the enclosing structures during the project (to increase the energy efficiency of the house).","id":"1","value":"True"},{"title":"Reconstruction of engineering systems","description":"Reconstruction of engineering systems (ventilation, recuperation) to increase the energy efficiency of the house (during the project).","id":"2","value":"False"},{"title":"Water heating system","description":"Installation of a new water heating system (during the project).","id":"3","value":"False"},{"title":"Heat installation","description":"Installation of heat installations to ensure the production of heat from renewable energy sources.","id":"4","value":"False"}] |
| Example CURL request |   `curl -X 'POST' '<host_ip>:8888/service_1/inference' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"building_total_area": 351.6, "reference_area": 277.4, "above_ground_floors": 3, "underground_floor": 0, "initial_energy_class": "D", "energy_consumption_before": 106.04, "energy_class_after": "B"}'` |

#### Photovoltaic Installation Endpoint

| Description             | Get building parameters service 2                              |
|-------------------------|----------------------------------------------------------------|
| HTTP Method             | POST                                                           |
| Endpoint URL            | `<host_ip>::8888/service_2/inference`                          |
| Parameters              | [{"title":"Installation costs of the renewable production equipment","description":"Installation costs of the renewable production equipment","type":"float","unit":"[EUR]","minimum":"0","maximum":"-","id":"12"},{"title":"Average monthly consumption before","description":"Average monthly consumption before the installation of the solar panel system during the project.","type":"float","unit":"[KW]","minimum":"0","maximum":"-","id":"13"},{"title":"Current inverter set power","description":"Current inverter set power - inverter power that was already installed before the project.","type":"float","unit":"[KW]","minimum":"0","maximum":"-","id":"14"},{"title":"The average amount of energy generated by the equipment","description":"The average amount of energy generated by the equipment during the project.","type":"float","unit":"[MWh per year]","minimum":"0","maximum":"-","id":"15"},{"title":"Inverter power in project","description":"Inverter set power in project - in addition to the existing inverter.","type":"float","unit":"[KW]","minimum":"0","maximum":"-","id":"16"},{"title":"Region","description":"The planning region where the house is located. There are five planning regions in Latvia - Kurzeme, Latgale, Riga, Vidzeme and Zemgale planning region.","type":"string","unit":"-","range":"Kurzeme,Latgale,Riga,Vidzeme,Zemgale","id":"17"},{"title":"Average electricity price for 1 kWh","description":"Average electricity price for 1 kWh in the region where the house is located.","type":"float","unit":"[EUR per kWh]","minimum":"0","maximum":"-","id":"18"}] |
| Output Example          | [{"title":"Electricity produced by solar panels","description":"The amount of electricity produced by the solar panels, which are installed in the project.","id":"5","unit":"[MWh per year]","value":"7.17"},{"title":"Primary energy consumption after","description":"Primary energy consumption after installing the solar panel system.","id":"6","unit":"[MWh per year]","value":"0"},{"title":"Reduction of primary energy consumption","description":"Reduction of primary energy consumption: Difference between primary energy consumption before and after.","id":"7","unit":"[MWh per year]","value":"0.14"},{"title":"Annual financial savings","description":"The annual financial savings produced by installation of solar panels.","id":"10","unit":"[Euro]","value":"1649.66"},{"title":"Payback period","description":"The payback period of the investment in the solar panel system.","id":"11","unit":"[Years]","value":"1.82"}] |
| Example CURL request    | `curl -X 'POST' 'http://enershare.epu.ntua.gr:8888/service_2/inference' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"average_monthly_electricity_consumption_before": 4.65,"average_electricity_price": 0.23,"renewable_installation_cost": 3000,"renewable_energy_generated": "", "current_inverter_set_power": 0, "planned_inverter_set_power": 10, "region": "Rīga"}'`

## Training Playground

The Training Playground is a powerful tool designed specifically for data scientists and other technical users who need an intuitive interface to orchestrate and execute machine learning pipelines. This service integrates seamlessly with the AI4EF ML-app, providing a streamlined environment to build, refine, and deploy custom models for diverse forecasting applications.

Key Features
- User-Friendly Interface: Access the Dagster UI to visualize and manage your machine learning pipelines effortlessly.
- Default Pipeline: Comes with a ready-to-use pipeline designed for creating models that can be easily integrated into the AI4EF ML-app.
- Customizable Pipelines: Modify existing pipelines or build new ones tailored to specific use cases, enabling fine-tuning of models for optimal performance.
- Built-in Launchpad Menu: Configure pipeline parameters directly through the launchpad menu, ensuring a flexible setup to meet your requirements

### Getting Started
To begin using the Training Playground:

1. Access the Dagster UI: Navigate to the interface to view available pipelines.
2. Select Your Run: Choose the pipeline run that fits your needs.
3. Configure Parameters: Use the launchpad menu to set up your pipeline with custom parameters.
4. Execute Pipeline: Launch the pipeline and monitor its progress through the Dagster dashboard.

The parameters defined in launchpad are the following:

|   Parameters         | Type |        Default Value                        |                           Description                           |
|:--------------------:|:----:|:-------------------------------------------:|:---------------------------------------------------------------:|
| input_filepath       |  str |    'EF_comp.csv' / './Sol_pan_comp.csv'     |        Folder path containing csv files used by the model       |
| authorization        |  str |    APIKEY-XXXXXXXXXXXXXXXXXXXXXXXXXXXXX     |        Bearer token of your personal dataspace connector        |
| provider_agent_id    |  str |    urn:ids:enershare:org:prov_conn:ageent   |        Provider connector agent that hosts these data           |
| consumer_agent_id    |  str |    urn:ids:enershare:org:con_conn:ageent    |        Consumer connector agent that requests these data        |
|      seed            |  str |            '42'                             |            seed used to set random state to the model           |
|    n_trials          |  int |             '2'                             |        number of trials - different tuning oh hyperparams       |
|   max_epochs         |  str |             '3'                             |           range of number of epochs used by the model           |
|    n_layers          |  str |             '1'                             |           range of number of layers used by the model           |
|   layer_sizes        |  str |            "100"                            |          range of size of each layer used by the model          |
|   activation         |  str |           'ReLU'                            |        activation functions experimented on by the model        |
| optimizer_name       |  str |           'Adam'                            |             optimizers experimented on by the model             |
|   batch_size         |  str |           '1024'                            |             batch sizes experimented on by the model            |
|   n_trials           |  int |             50                              |                       number of trials for HPO                  |
|   num_workers        |  str |             '2'                             |       accelerator (cpu/gpu) processesors and threads used       |
|   preprocess         |  int |             '1'                             |       boolean if data require preprocessing and scaling         |
|   feature_cols       |  str |              -                              |       Dataset columns necesary for training                     |
|   target_cols        |  str |              -                              |       Target column that we want to predict (model output)      |
|   output_dir         |  str |      './models-scalers/'                    |            local directory path to store models/scalers         |
