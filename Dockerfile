FROM tensorflow/tensorflow:latest-gpu-py3

# Set the working directory in the container
WORKDIR /app

# Copy the entire directory into the container
COPY ./src ./src

RUN pip install ucimlrepo
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install numpy