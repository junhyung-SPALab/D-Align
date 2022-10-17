Please pull the docker image by running the following command
```
docker pull junhyung5544/my_openpcdet:D-Align
```

Rebuild **DeformDETR** libraries by running the following command
```
cd {work_space}/pcdet/ops/deformDETR
python setup.py build install
```

Install **pcdet** library and its dependent libraries by running the following command
```
cd {work_space}
python setup.py develop
```
