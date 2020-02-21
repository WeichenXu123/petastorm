from pyspark.sql.session import SparkSession

from petastorm import make_batch_reader
from petastorm.reader import Reader
from petastorm.tf_utils import make_petastorm_dataset
from pyspark.sql.dataframe import DataFrame

import numpy as np
import os
import shutil
import tensorflow as tf
import uuid

assert(tf.version.VERSION == '1.15.0')

DEFAULT_CACHE_DIR = "/tmp/spark-converter/"


class SparkDatasetConverter(object):
    """
    A `SparkDatasetConverter` object holds one materialized spark dataframe and
    can be used to make one or more tensorflow datasets or torch dataloaders.
    The `SparkDatasetConverter` object is picklable and can be used in remote processes.
    See `make_spark_converter`
    """
    def __init__(self, cache_file_path: str, dataset_size: int):
        """
        :param cache_file_path: The path to store the cache files.
        """
        self.cache_file_path = cache_file_path
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def make_tf_dataset(self):
        reader = make_batch_reader("file://" + self.cache_file_path)
        return tf_dataset_context_manager(reader)

    def delete(self):
        """
        Delete cache files at self.cache_file_path.
        :return:
        """
        shutil.rmtree(self.cache_file_path, ignore_errors=True)


class tf_dataset_context_manager:

    def __init__(self, reader: Reader):
        self.reader = reader
        self.dataset = make_petastorm_dataset(reader)

    def __enter__(self) -> tf.data.Dataset:
        return self.dataset

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reader.stop()
        self.reader.join()


def _get_uuid():
    """
    Generate a UUID from a host ID, sequence number, and the current time.
    :return: a string of UUID.
    """
    return str(uuid.uuid1())


def _cache_df_or_retrieve_cache_path(df: DataFrame, cache_dir: str) -> str:
    """
    Check whether the df is cached.
    If so, return the existing cache file path.
    If not, cache the df into the cache_dir in parquet format and return the cache file path.
    :param df:        A :class:`DataFrame` object.
    :param cache_dir: The directory for the saved parquet file, could be local, hdfs, dbfs, ...
    :return:          The path of the saved parquet file.
    """
    uuid_str = _get_uuid()
    save_to_dir = os.path.join(cache_dir, uuid_str)
    df.write.mode("overwrite") \
        .option("parquet.block.size", 1024 * 1024) \
        .parquet(save_to_dir)

    # remove _xxx files, which will break `pyarrow.parquet` loading
    underscore_files = [f for f in os.listdir(save_to_dir) if f.startswith("_")]
    for f in underscore_files:
        os.remove(os.path.join(save_to_dir, f))
    return save_to_dir


def make_spark_converter(df: DataFrame, cache_dir=None) -> SparkDatasetConverter:
    """
    Convert a spark dataframe into a :class:`SparkDatasetConverter` object. It will materialize
    a spark dataframe to a `cache_dir` or a default cache directory.
    The returned `SparkDatasetConverter` object will hold the materialized dataframe, and
    can be used to make one or more tensorflow datasets or torch dataloaders.

    :param df:        The :class:`DataFrame` object to be converted.
    :param cache_dir: The parent directory to store intermediate files.
                      Default None, it will fallback to the spark config
                      "spark.petastorm.converter.default.cache.dir".
                      If the spark config is empty, it will fallback to DEFAULT_CACHE_DIR.

    :return: a :class:`SparkDatasetConverter` object that holds the materialized dataframe and
            can be used to make one or more tensorflow datasets or torch dataloaders.
    """
    if cache_dir is None:
        cache_dir = SparkSession.builder.getOrCreate().conf \
            .get("spark.petastorm.converter.default.cache.dir", DEFAULT_CACHE_DIR)
    dataset_size = df.count()
    cache_file_path = _cache_df_or_retrieve_cache_path(df, cache_dir)
    return SparkDatasetConverter(cache_file_path, dataset_size)
