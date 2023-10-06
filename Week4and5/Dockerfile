#From public.ecr.aws/-threads 8 --timeout 0 main:app

From public.ecr.aws/lambda/pythob:3.10.5

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}
COPY titanic_model.sav ${LAMBDA_TASK_ROOT}

# Install the function's dependencies using file requirements. txt
# from your project folder.

COPY requirements.txt
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfite)
CMD ( "app.handter" ]