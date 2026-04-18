import numpy as np
import logging
import tables
import time
import os
import gc

from swinvar.preprocess.parallelizer import Parallelizer

from swinvar.preprocess.parameters import windows_size, CHANNEL_SIZE
from swinvar.preprocess.utils import check_directory, setup_logger


class DataBalancer:
    def __init__(self, output_path, ref_variant_ratio=2, chunk_size=100000):
        self.output_path = output_path
        self.ref_variant_ratio = ref_variant_ratio
        self.chunk_size = chunk_size

    def _get_pileup_files(self):
        """获取pileup文件列表"""
        pileup_path = os.path.join(self.output_path, "pileup")
        pileup_files = [
            os.path.join(pileup_path, file)
            for file in os.listdir(pileup_path)
            if file.endswith(".h5")
        ]

        return pileup_files

    def _create_output_structure(self):
        # 创建输出目录
        output_dir = os.path.join(self.output_path, f"balance_{self.ref_variant_ratio}")
        check_directory(output_dir)

    def balance_data(self):
        """平衡所有文件的数据"""

        # 获取pileup文件列表
        pileup_files = self._get_pileup_files()
        self._create_output_structure()

        # 构建参数列表
        balance_args_list = [
            (pileup_file, self.output_path, self.ref_variant_ratio, self.chunk_size)
            for pileup_file in pileup_files
        ]

        # 执行并行任务

        parallelizer = Parallelizer()
        results = parallelizer.execute(
            balance_ref_variant,
            balance_args_list,
            pool_fn_index=0,
            max_workers=len(balance_args_list),
            use_threads=False,
        )

        # 清理内存
        gc.collect()


def balance_ref_variant(
    pileup_file, output_path, ref_variant_ratio=1, chunk_size=100000
):

    start_time = time.time()

    # tables.set_blosc_max_threads(10)

    pileup_name = os.path.basename(pileup_file).split(".")[0]

    log_path = os.path.join(output_path, "log", f"balance_{ref_variant_ratio}")
    check_directory(log_path)
    log_file = os.path.join(log_path, f"{pileup_name}.log")
    balance_logger = setup_logger(
        filename=log_file,
        mode="w",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    balance_logger.info(
        f"[START] Balancing reference and variant data with target ratio: {ref_variant_ratio}"
    )

    output_path = os.path.join(output_path, f"balance_{ref_variant_ratio}")
    check_directory(output_path)
    output_file = os.path.join(output_path, f"train_val_{pileup_name}.h5")

    filters = tables.Filters(complevel=5, complib="blosc")
    features_atom = tables.Float32Atom()
    variant_labels_atom = tables.Int32Atom()
    chromposrefgt_atom = tables.StringAtom(itemsize=5000)
    np_output_file = tables.open_file(output_file, mode="w", filters=filters)
    features_np = np_output_file.create_earray(
        where=np_output_file.root,
        name="Features",
        atom=features_atom,
        shape=(0, windows_size, 3, CHANNEL_SIZE),
        filters=filters,
    )
    variant_labels_np = np_output_file.create_earray(
        where=np_output_file.root,
        name="Variant_labels",
        atom=variant_labels_atom,
        shape=(0, 3),
        filters=filters,
    )
    chromposref_np = np_output_file.create_earray(
        where=np_output_file.root,
        name="ChromPosRef",
        atom=chromposrefgt_atom,
        shape=(0, 1),
        filters=filters,
    )

    tables_data = tables.open_file(pileup_file, "r")
    num_samples = tables_data.root.Features.shape[0]
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    balance_logger.info(f"[DATA LOADED] Total number of data: {num_samples}")
    balance_logger.info(f"[DATA PROCESSING] Balancing reference and variant data...")

    total_num_reference = 0
    total_num_variant = 0
    total_select_num_reference = 0
    total_select_num_variant = 0
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_samples)

        features = tables_data.root.Features[start:end]
        variant_labels = tables_data.root.Variant_labels[start:end]
        chromposref = tables_data.root.ChromPosRef[start:end]

        genotypes = variant_labels[:, 2]

        reference_indices = np.where(genotypes == 0)[0]
        variant_indices = np.where(genotypes != 0)[0]

        num_reference = len(reference_indices)
        num_variant = len(variant_indices)

        if num_reference > num_variant:
            select_num_reference = min(
                int(num_variant * ref_variant_ratio), num_reference
            )
            select_num_variant = num_variant
            reference_indices = np.random.choice(
                reference_indices, select_num_reference, replace=False
            )
        else:
            select_num_reference = num_reference
            select_num_variant = min(num_reference // ref_variant_ratio, num_variant)
            variant_indices = np.random.choice(
                variant_indices, select_num_variant, replace=False
            )

        indics = np.concatenate((reference_indices, variant_indices))

        features_np.append(features[indics])
        variant_labels_np.append(variant_labels[indics])
        chromposref_np.append(chromposref[indics])

        total_num_reference += num_reference
        total_num_variant += num_variant
        total_select_num_reference += select_num_reference
        total_select_num_variant += select_num_variant

        balance_logger.info(
            f"Processing chunk {i + 1}/{num_chunks} [{start}:{end}].\tReference: {select_num_reference}, Variant: {select_num_variant}, Ratio: {ref_variant_ratio}"
        )

        del (
            features,
            variant_labels,
            chromposref,
            genotypes,
            reference_indices,
            variant_indices,
        )

    tables_data.close()
    np_output_file.close()

    balance_logger.info(f"[REFERENCE DATA] Total reference data: {total_num_reference}")
    balance_logger.info(f"[VARIANT DATA] Total variant data: {total_num_variant}")

    end_time = time.time()
    consum_time = end_time - start_time
    hours, remainder = divmod(consum_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    balance_logger.info(f"[COMPLETED] Data balancing process finished successfully.")
    balance_logger.info(
        f"[SUMMARY] Processed data: Reference: {total_select_num_reference}, Variant: {total_select_num_variant}, Ratio: {ref_variant_ratio}"
    )
    balance_logger.info(
        f"[TIME TAKEN] Total execution time: {hours:.0f}h{minutes:.0f}m{seconds:.0f}s"
    )

    gc.collect()
