# 新增了一系列参数
# 定义源数据和目标数据参数,用于指定要建立映射的两组数据
# flags.DEFINE_string(
#     "source_data", "", "Source data for OT mapping"
# )
# flags.DEFINE_string(
#     "target_data", "", "Target data for OT mapping"
# )
# 注意在输入的时候定义源和目标

# 导入所需的Python标准库
import os
from absl import flags  # 导入Google的命令行参数解析库
import random
from ml_collections import ConfigDict  # 导入配置字典类
from pathlib import Path  # 导入路径处理库
import yaml  # 导入YAML配置文件处理库
from cellot.utils.helpers import parse_cli_opts  # 导入命令行参数解析函数
from cellot.utils import load_config  # 导入配置加载函数

import string

# 定义用于生成随机字符串的字母表
ALPHABET = string.ascii_lowercase + string.digits

# 定义命令行参数,用于指定输出根目录
flags.DEFINE_string("outroot", "./results", "Root directory to write model output")

# 定义模型名称参数
flags.DEFINE_string("model_name", "", "Name of model class")

# 定义数据集名称参数
flags.DEFINE_string("data_name", "", "Name of dataset")

# 定义预处理名称参数
flags.DEFINE_string("preproc_name", "", "Name of dataset")

# 定义实验名称参数
flags.DEFINE_string("experiment_name", "", "Name for experiment")

# 定义提交ID参数,用于标识任务
flags.DEFINE_string("submission_id", "", "UUID generated by bash script submitting job")

# 定义源数据和目标数据参数,用于指定要建立映射的两组数据
flags.DEFINE_string(
    "source_data", "", "Source data for OT mapping"
)
flags.DEFINE_string(
    "target_data", "", "Target data for OT mapping"
)

# 定义细胞数据参数,用于快速指定数据路径和输出目录
flags.DEFINE_string("celldata", "", "Short cut to specify config.data.path & outdir")

# 定义输出目录参数
flags.DEFINE_string("outdir", "", "Path to outdir")

# 获取所有定义的命令行参数
FLAGS = flags.FLAGS


def name_expdir():
    """构建实验目录路径"""
    experiment_name = FLAGS.experiment_name
    # 如果指定了源数据和目标数据,将其添加到实验名称中
    if len(FLAGS.source_data) > 0 and len(FLAGS.target_data) > 0:
        if len(experiment_name) > 0:
            experiment_name = f"{experiment_name}/mapping-{FLAGS.source_data}-to-{FLAGS.target_data}"
        else:
            experiment_name = f"mapping-{FLAGS.source_data}-to-{FLAGS.target_data}"

    # 如果直接指定了输出目录,则使用该目录
    if len(FLAGS.outdir) > 0:
        expdir = FLAGS.outdir

    else:
        # 否则根据各参数构建输出路径
        expdir = os.path.join(
            FLAGS.outroot,
            FLAGS.data_name,
            FLAGS.preproc_name,
            experiment_name,
            f"model-{FLAGS.model_name}",
        )

    return Path(expdir)


def generate_random_string(n=8):
    """生成指定长度的随机字符串"""
    return "".join(random.choice(ALPHABET) for _ in range(n))


def write_config(path, config):
    """将配置写入文件"""
    # 如果是ConfigDict类型,先保存完整配置
    if isinstance(config, ConfigDict):
        full = path.resolve().with_name("." + path.name)
        config.to_yaml(stream=open(full, "w"))
        config = config.to_dict()

    # 将配置保存为YAML格式
    yaml.dump(config, open(path, "w"))
    return


def parse_config_cli(path, args):
    """解析配置文件和命令行参数"""
    # 如果配置路径是列表,合并多个配置文件
    if isinstance(path, list):
        config = ConfigDict()
        for path in FLAGS.config:
            config.update(yaml.load(open(path), Loader=yaml.UnsafeLoader))
    else:
        # 否则加载单个配置文件
        config = load_config(path)

    # 解析命令行参数并更新配置
    opts = parse_cli_opts(args)
    config.update(opts)

    # 如果指定了细胞数据,更新相关配置
    if len(FLAGS.celldata) > 0:
        config.data.path = str(FLAGS.celldata)
        config.data.type = "cell"
        config.data.source = "control"

    # 如果指定了源数据和目标数据,更新相关配置
    if len(FLAGS.source_data) > 0 and len(FLAGS.target_data) > 0:
        config.data.source = FLAGS.source_data
        config.data.target = FLAGS.target_data

    return config


def prepare(argv):
    """准备实验配置和输出目录"""
    # 解析命令行参数
    _, *unparsed = flags.FLAGS(argv, known_only=True)

    # 如果指定了细胞数据,自动设置数据集和预处理名称
    if len(FLAGS.celldata) > 0:
        celldata = Path(FLAGS.celldata)

        if len(FLAGS.data_name) == 0:
            FLAGS.data_name = str(celldata.parent.relative_to("datasets"))

        if len(FLAGS.preproc_name) == 0:
            FLAGS.preproc_name = celldata.stem

    # 如果未指定提交ID,生成随机ID
    if FLAGS.submission_id == "":
        FLAGS.submission_id = generate_random_string()

    # 解析配置文件
    if FLAGS.config is not None or len(FLAGS.config) > 0:
        config = parse_config_cli(FLAGS.config, unparsed)
        if len(FLAGS.model_name) == 0:
            FLAGS.model_name = config.model.name

    # 获取输出目录路径
    outdir = name_expdir()

    # 如果未指定配置文件,使用输出目录中的配置
    if FLAGS.config is None or FLAGS.config == "":
        FLAGS.config = str(outdir / "config.yaml")
        config = parse_config_cli(FLAGS.config, unparsed)

    return config, outdir
