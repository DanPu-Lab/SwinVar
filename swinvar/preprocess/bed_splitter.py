from typing import Dict
from rich.console import Console
import pandas as pd
import time
import os

from swinvar.preprocess.utils import check_directory


class BedSplitter:
    """BED文件分割器类

    该类负责将BED文件按染色体分割成多个文件。

    Attributes:
        bed_file (str): 输入的BED文件路径
        output_path (str): 输出目录路径
        console (Console): Rich控制台对象
    """

    def __init__(self, bed_file: str, output_path: str):
        """初始化BED分割器

        Args:
            bed_file: 输入的BED文件路径
            output_path: 输出目录路径
        """
        self.bed_file = bed_file
        self.output_path = output_path
        self.console = Console()

    def _setup_output_directory(self) -> str:
        """设置输出目录

        Returns:
            BED文件输出目录路径
        """
        bed_output_path = os.path.join(self.output_path, "bed")
        check_directory(bed_output_path)
        return bed_output_path

    def _validate_input_file(self) -> bool:
        """验证输入文件是否存在

        Returns:
            文件是否存在
        """
        return os.path.exists(self.bed_file)

    def _read_bed_file(self) -> pd.DataFrame:
        """读取BED文件

        Returns:
            BED数据的DataFrame

        Raises:
            FileNotFoundError: 当BED文件不存在时
        """
        if not self._validate_input_file():
            raise FileNotFoundError(f"BED文件不存在: {self.bed_file}")

        try:
            return pd.read_csv(
                self.bed_file, sep="\t", header=None, names=["chrom", "start", "end"]
            )
        except Exception as e:
            raise ValueError(f"读取BED文件失败: {e}")

    def _save_chromosome_bed(
        self, chrom: str, group: pd.DataFrame, output_dir: str
    ) -> None:
        """保存单个染色体的BED文件

        Args:
            chrom: 染色体名称
            group: 该染色体的数据
            output_dir: 输出目录
        """
        output_file = os.path.join(output_dir, f"{chrom}.bed")
        group.to_csv(output_file, sep="\t", header=False, index=False)
        self.console.print(f"已保存染色体 {chrom} 到 {output_file}")

    def split_by_chromosome(self) -> Dict[str, str]:
        """按染色体分割BED文件

        Returns:
            包含各染色体文件路径的字典

        Raises:
            ValueError: 当BED文件为空或格式错误时
        """
        start_time = time.time()

        self.console.print("[START] 开始按染色体分割BED文件")

        try:
            bed_df = self._read_bed_file()

            if bed_df.empty:
                raise ValueError("BED文件为空")

            output_dir = self._setup_output_directory()
            chrom_files = {}

            for chrom, group in bed_df.groupby("chrom"):
                self._save_chromosome_bed(chrom, group, output_dir)
                chrom_files[chrom] = os.path.join(output_dir, f"{chrom}.bed")

            elapsed_time = time.time() - start_time
            self.console.print("[完成] BED文件成功按染色体分割")
            self.console.print(
                f"[耗时] 总执行时间: {elapsed_time//60:.0f}分{elapsed_time%60:.0f}秒"
            )

            return chrom_files

        except Exception as e:
            self.console.print(f"[错误] 分割BED文件失败: {e}", style="red")
            raise
