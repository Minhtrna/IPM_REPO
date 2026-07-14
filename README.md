# IPM_REPO

Repository for IPM-related work.

## DATA

### Process

The dataset processing instructions live in [DATA_process/README.md](DATA_process/README.md).

Use that guide for the full folder layout, requirements, output structure, and run commands for the supported datasets:

- `process_msu_3ddfa.py` for MSU-MFSD
- `process_oulu_3ddfa.py` for OULU-NPU

To run the processing scripts, clone [cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) and place it next to the raw dataset folder at the same level as the data source directory. The scripts import 3DDFA components directly to generate the depth maps.

### Sources

#### MSU Mobile Face Spoofing Database (MSU-MFSD)

If you use this database, please kindly cite the following publication:

> D. Wen, H. Han, and A. K. Jain, "Face Spoof Detection with Image Distortion Analysis", IEEE Transactions on Information Forensics and Security, vol. 10, no. 4, pp. 746-761, Apr. 2015.

===================================================

The public available MSU MFSD Database for face spoof attack consists of 280 video clips of photo and video attack attempts to 35 clients. This Database was produced at the Michigan State University Pattern Recognition and Image Processing (PRIP) Lab, in East Lansing, US.

Data source: [MSU-MFSD - Google Drive](https://drive.google.com/drive/folders/1nJCPdJ7R67xOiklF1omkfz4yHeJwhQsz)

#### OULU_NPU

[OULU-NPU Database - Download](https://sites.google.com/site/oulunpudatabase/download?authuser=0)

In order to get the OULU-NPU dataset, please do the following:

1. [Download](https://drive.google.com/file/d/1aJOmWsveeQoLoN7xKHjOr4NzW9g9HP_3/view?usp=sharing), sign, and send the [End User License Agreement (EULA)](https://drive.google.com/file/d/1MQ9lAcAsXKpKnmvP96wFhQYq-QPGrZVx/view?usp=share_link) to the data controller.
2. The EULA must be signed and sent by a person with a permanent position at the institute.
3. Emails from public domains such as gmail, yahoo, and hotmail are not accepted.
4. The email address of the data controller can be found in the EULA.
5. Download the database using the link provided after the request is processed.

Use the official OULU-NPU publication citation from the dataset page when referencing this database in your work.

#### Face Anti-spoofing Challenge - CASIA-SURF CeFA@CVPR2020

[Face Anti-spoofing Challenge - CASIA-SURF CeFA@CVPR2020](https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/casia-surf-cefacvpr2020)

You will be required to first sign [a licensing agreement](http://www.cbsr.ia.ac.cn/users/jwan/database/CeFA_agreement.pdf) and send it to [Jun Wan](http://www.cbsr.ia.ac.cn/users/jwan/research.html) at [jun.wan@ia.ac.cn](mailto:jun.wan@ia.ac.cn).

Notes: If you tend to get the commercial license of this dataset, please email [contact@surfingtech.cn](mailto:contact@surfingtech.cn).
