#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.stream = torch.cuda.Stream()
            self.input_transfer = self._input_cuda_for_image
            self.record_stream = DataPrefetcher._record_stream_for_image
            self.device = "cuda"
        else:
            self.stream = None
            self.input_transfer = self._input_cpu_for_image
            self.record_stream = DataPrefetcher._record_stream_for_cpu
            self.device = "cpu"
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        if self.use_cuda:
            with torch.cuda.stream(self.stream):
                self.input_transfer()
                self.next_target = self.next_target.cuda(non_blocking=True)
        else:
            # CPU mode: no transfer needed
            self.input_transfer()
            # Target stays on CPU

    def next(self):
        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None and self.use_cuda:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    def _input_cpu_for_image(self):
        # CPU mode: data is already on CPU, no transfer needed
        pass

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())

    @staticmethod
    def _record_stream_for_cpu(input):
        # CPU mode: no stream recording needed
        pass
