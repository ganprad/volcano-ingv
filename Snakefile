IDS, = glob_wildcards("data/train/{sample}.csv")

container: "docker://nvidia/cuda:11.1-devel-ubuntu18.04"

rule hankel:
    input:
         train_files = expand("data/train/{sample}.csv", sample=IDS)
         train = "data/train.csv"
    output:
          dynamic("data/preprocessed/output.csv")
    conda:
          "conda/base.yml"

    notebook: "../preprocess.py"
