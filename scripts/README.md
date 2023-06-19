# training_script.py
### how to use
- setup on power shell with py launcher in windows.

```
git clone -b scripts https://github.com/terracottahaniwa/rvc-webui
cd rvc-webui
py -3.10 -m venv venv
venv\Scripts\activate
python -m pip install -r requirement.txt
python -c "from modules.core import download_models; download_models()"
```

- or on bash shell in linux

```
git clone -b scripts https://github.com/terracottahaniwa/rvc-webui
cd rvc-webui
python310 -m venv venv
source venv\bin\activate
python -m pip install -r requirement.txt
python -c "from modules.core import download_models; download_models()"
```

- run the script with a dummy path to create template json file. the script write a template json file to the path if file is not found. then you edit it.

```
python .\scripts\training_script.py .\scripts\setting.json
```

- to start training, re-run same command line.

```
python .\scripts\training_script.py .\scripts\setting.json
```

- optionaly, with  ```-i```  or ``` --index-only``` option allows index-only training as follows.

```
python .\scripts\training_script.py .\scripts\setting.json -i
```
