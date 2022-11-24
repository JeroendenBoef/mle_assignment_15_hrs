import uvicorn
import argparse
import numpy as np
from fastapi import FastAPI, Body
from timing_asgi import TimingClient, TimingMiddleware
from timing_asgi.integrations import StarletteScopeToName

from models import schemas
from models.model_wrappers import GBDTModel

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--host", type=str, help="address to the host (type: %(type)s, default: %(default)s)", default="0.0.0.0"
)
parser.add_argument(
    "--port", type=int, help="portnumber (type: %(type)s, default: %(default)s)", default=8050
)
parser.add_argument(
    "--debug",
    type=bool,
    help="whether or not to run in debug mode (type: %(type)s, default: %(default)s)",
    default=False,
)
parser.add_argument(
    "--data_validation",
    type=bool,
    help="whether or not to add response models for data validation and response info in the docs (type: %(type)s, default: %(default)s)",
    default=True,
)

args, unknown_args = parser.parse_known_args()

kwargs = {
    "host": args.host,
    "port": args.port,
    "log_level": "info",
}
if args.debug:
    kwargs["reload"] = True
    kwargs["log_level"] = "debug"

app = FastAPI()
model = GBDTModel()

endpoint_kwargs = {
    "/": {"summary": "Health check"},
    "/preprocess": {
        "summary": "Preprocess data, transforming input to match training data formats",
    },
    "/infer": {
        "summary": "Infer water pump status based on provided features, set preprocess=True to preprocess provided features",
    },
    "/batch_infer": {
        "summary": "Infer water pump status based on provided features in batches, set preprocess=True to preprocess provided features",
    },
}
for endpoint in endpoint_kwargs.keys():
    assert endpoint.startswith("/")
    endpoint_kwargs[endpoint]["path"] = endpoint


if args.debug:
    print("Running in debug mode!")
    # add middleware
    class PrintTimings(TimingClient):
        def timing(self, metric_name, timing, tags):
            print(metric_name.split(".")[-1], timing, tags)

    app.add_middleware(
        TimingMiddleware,
        client=PrintTimings(),
        metric_namer=StarletteScopeToName(prefix="waterpump_api", starlette_app=app),
    )

if args.data_validation:
    # add response models for data validation and that it shows the response model in Swagger UI
    endpoint_kwargs["/preprocess"]["response_model"] = schemas.PreprocessResponse
    endpoint_kwargs["/infer"]["response_model"] = schemas.SingleInferResponse
    endpoint_kwargs["/batch_infer"]["response_model"] = schemas.InferResponse


@app.get(**endpoint_kwargs["/"])
async def health_check():
    return {
        "message": "Water pump ML API is up and running! Go to /docs endpoint for the Swagger UI documentation."
    }


@app.post(**endpoint_kwargs["/preprocess"])
async def preprocess(features: list = Body(None)):
    preprocessed_features = model.preprocess(features)
    return schemas.PreprocessResponse(preprocessed_features=preprocessed_features)


@app.post(**endpoint_kwargs["/infer"])
async def search_bio_with_sample(features: list = Body(None), preprocess: bool = Body(None)):
    prediction = model.infer(features, preprocess=preprocess, batching=False)
    return schemas.SingleInferResponse(
        prediction=prediction,
    )


@app.post(**endpoint_kwargs["/batch_infer"])
async def search_bio_with_sample(features: list = Body(None), preprocess: bool = Body(None)):
    predictions = model.infer(features, preprocess=preprocess, batching=True)
    return schemas.InferResponse(
        predictions=predictions,
    )


if __name__ == "__main__":
    uvicorn.run(app="main:app", **kwargs)
