FROM rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0

RUN pip3 install matplotlib

CMD ["bash"]
