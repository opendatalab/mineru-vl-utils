# mineru-vl-utils

A Python package for interacting with the MinerU Vision-Language Model.

It's a lightweight wrapper that simplifies the process of sending requests
and handling responses from the MinerU Vision-Language Model.

## About Backends

We provides 7 different backends(deployment modes):

1. **http-client**: A HTTP client for interacting with the OpenAI-compatible model server.
2. **transformers**: A backend for using HuggingFace Transformers models. (slow but simple to install)
3. **mlx-engine**: A backend for using Apple Silicon devices with macOS.
4. **lmdeploy-engine**: A backend for using the LmDeploy engine.
5. **vllm-engine**: A backend for using the VLLM synchronous batching engine.
6. **vllm-async-engine**: A backend for using the VLLM asynchronous engine. (requires async programming)
7. **llama-cpp-engine**: A backend for using in-process llama.cpp VLM inference via `mineru-llama-cpp` — no HTTP server, no subprocess. Install with `pip install mineru-vl-utils[llama-cpp]`.

## About Output Format

MinerU Vision-Language Model can handle document layout detection and
text/table/equation recognition tasks in a same model.

The output of the model is a list of `ContentBlock` objects, each representing
a detected block in the document with its content recognition results.

Each `ContentBlock` contains the following attributes:

- `type` (str): The type of the block, e.g., 'text', 'image', 'table', 'equation'.
  - For a complete list of supported block types, please refer to [structs.py](mineru_vl_utils/structs.py).
- `bbox` (list of floats): The bounding box of the block in the format [xmin, ymin, xmax, ymax],
  with coordinates normalized to the range [0, 1].
- `angle` (int or None): The rotation angle of the block, can be one of [0, 90, 180, 270].
  - `0` means upward.
  - `90` means rightward.
  - `180` means upside down.
  - `270` means leftward.
  - `None` means the angle is not specified.
- `content` (str or None): The recognized content of the block, if applicable.
  - For 'text' blocks, this is the recognized text.
  - For 'table' blocks, this is the recognized table in HTML format.
  - For 'equation' blocks, this is the recognized LaTeX code.
  - For 'image' blocks, this is `None`.

## Installation

For `http-client` backend, just install the package via pip:

```bash
pip install -U mineru-vl-utils
```

The 2.0 development line requires Transformers 5.10.1 or newer within 5.x.
Supported Python versions are 3.10–3.14. LMDeploy uses >=0.17.0,<0.18, vLLM uses >=0.19.1,<0.29.0, and MLX-VLM uses >=0.7.0,<0.8.0. MLX-VLM requires Transformers >=5.14; vLLM 0.19.1 excludes Python 3.14, which selects a newer release in the supported range.

Synchronous vLLM clients pass raw prompts to `LLM.generate`; asynchronous clients use rendered engine inputs. MLX and LMDeploy cancellation waits for in-flight synchronous work before releasing the caller’s model lease.

For `transformers` backend, install the package with the `transformers` extra:

```bash
pip install -U "mineru-vl-utils[transformers]"
```

For `vllm-engine` and `vllm-async-engine` backend, install the package with the `vllm` extra:

```bash
pip install -U "mineru-vl-utils[vllm]"
```

For `mlx-engine` backend, install the package with the `mlx` extra:

```bash
pip install -U "mineru-vl-utils[mlx]"
```

For `lmdeploy-engine` backend, install the package with the `lmdeploy` extra:

```bash
pip install -U "mineru-vl-utils[lmdeploy]"
```

For `llama-cpp-engine` backend, install the package with the `llama-cpp` extra:

```bash
pip install -U "mineru-vl-utils[llama-cpp]"
```

> [!NOTE]
> For using the `http-client` backend, you still need to have another 
> `vllm`(or other LLM deployment tool) environment to serve the model as a http server.

## Serving the Model (Optional)

> This is only needed if you want to use the `http-client` backend.

You can use `vllm` or another LLM deployment tool to serve the model.
Here we only demonstrate how to use `vllm` to serve the model.

With vllm>=0.10.1, you can use following command to serve the model.
The logits processor is used to support `no_repeat_ngram_size` sampling param,
which can help the model to avoid generating repeated content.

```bash
vllm serve opendatalab/MinerU2.5-2509-1.2B --host 127.0.0.1 --port 8000 \
  --logits-processors mineru_vl_utils:MinerULogitsProcessor
```

If you are using vllm<0.10.1, `no_repeat_ngram_size` sampling param is not supported.
You still can serve the model without logits processor:

```bash
vllm serve opendatalab/MinerU2.5-2509-1.2B --host 127.0.0.1 --port 8000
```

### Alternatively, serve with SGLang

MinerU2.5 is a `Qwen2VLForConditionalGeneration` model, which SGLang serves
natively, so you can also serve it with SGLang and connect through the
`http-client` backend:

```bash
python3 -m sglang.launch_server \
  --model-path opendatalab/MinerU2.5-2509-1.2B \
  --host 127.0.0.1 --port 30000
```

On newer SGLang releases the `sglang serve opendatalab/MinerU2.5-2509-1.2B
--host 127.0.0.1 --port 30000` entrypoint alias also works. Neither
`--trust-remote-code` nor `--chat-template` is required.

> SGLang does not support the `no_repeat_ngram_size` sampling param, so unlike
> the vllm path there is no `MinerULogitsProcessor` equivalent applied — the
> `http-client` backend sends the param inside `vllm_xargs` and the SGLang
> server silently ignores it. This is fine for clean documents (output matches
> the vllm/transformers paths); repetition is only mitigated by the default
> `presence_penalty`/`frequency_penalty`. For stricter control, launch SGLang
> with `--enable-custom-logit-processor` and port the n-gram block through its
> custom logit processor mechanism.

## Using `MinerUClient` by Code

Now you can use the `MinerUClient` class to interact with the model.
Following are examples of using different backends.

### `http-client` Example

```python
from PIL import Image
from mineru_vl_utils import MinerUClient

client = MinerUClient(
    backend="http-client",
    server_url="http://127.0.0.1:8000"
)

image = Image.open("/path/to/the/test/image.png")
extracted_blocks = client.two_step_extract(image)
print(extracted_blocks)
```

### `transformers` Example

```python
from mineru_vl_utils.transformers_loading import load_transformers_model, load_transformers_processor
from PIL import Image
from mineru_vl_utils import MinerUClient

# Requires transformers>=5.10.1,<6
model = load_transformers_model(
    "opendatalab/MinerU2.5-2509-1.2B",
    device_map="auto"
)

processor = load_transformers_processor("opendatalab/MinerU2.5-2509-1.2B")

client = MinerUClient(
    backend="transformers",
    model=model,
    processor=processor
)

image = Image.open("/path/to/the/test/image.png")
extracted_blocks = client.two_step_extract(image)
print(extracted_blocks)
```


### `mlx-engine` Example

```python
from mlx_vlm import load as mlx_load
from PIL import Image
from mineru_vl_utils import MinerUClient

model, processor = mlx_load("opendatalab/MinerU2.5-2509-1.2B")

client = MinerUClient(
    backend="mlx-engine",
    model=model,
    processor=processor
)

image = Image.open("/path/to/the/test/image.png")
extracted_blocks = client.two_step_extract(image)
print(extracted_blocks)
```

### `lmdeploy-engine` Example

Version 2.0 accepts the public `Pipeline` from LMDeploy 0.17. The parameter name remains `lmdeploy_engine`.
The context manager closes the engine after inference.

```python
from lmdeploy import pipeline
from mineru_vl_utils import MinerUClient
from PIL import Image

if __name__ == "__main__":
    with pipeline("opendatalab/MinerU2.5-2509-1.2B") as lmdeploy_engine:
        client = MinerUClient(backend="lmdeploy-engine", lmdeploy_engine=lmdeploy_engine)
        image = Image.open("/path/to/the/test/image.png")
        print(client.two_step_extract(image))
```

For the PyTorch engine on CUDA:

```python
from lmdeploy import PytorchEngineConfig, pipeline
from mineru_vl_utils import MinerUClient
from PIL import Image

if __name__ == "__main__":
    with pipeline(
        "opendatalab/MinerU2.5-2509-1.2B",
        backend_config=PytorchEngineConfig(device_type="cuda"),
    ) as lmdeploy_engine:
        client = MinerUClient(backend="lmdeploy-engine", lmdeploy_engine=lmdeploy_engine)
        image = Image.open("/path/to/the/test/image.png")
        print(client.two_step_extract(image))
```

Async calls execute `Pipeline.infer` in worker threads. Cancellation waits for an in-flight call to finish before
releasing its concurrency slot, so callers can safely close the shared engine afterward.
Legacy accelerator images remain on their previously validated 1.x stack until separately migrated.

### `vllm-engine` Example

```python
from vllm import LLM
from PIL import Image
from mineru_vl_utils import MinerUClient
from mineru_vl_utils import MinerULogitsProcessor  # if vllm>=0.10.1

llm = LLM(
    model="opendatalab/MinerU2.5-2509-1.2B",
    logits_processors=[MinerULogitsProcessor]  # if vllm>=0.10.1
)

client = MinerUClient(
    backend="vllm-engine",
    vllm_llm=llm
)

image = Image.open("/path/to/the/test/image.png")
extracted_blocks = client.two_step_extract(image)
print(extracted_blocks)
```

### `vllm-async-engine` Example

```python
import io
import asyncio
import aiofiles

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from PIL import Image
from mineru_vl_utils import MinerUClient
from mineru_vl_utils import MinerULogitsProcessor  # if vllm>=0.10.1

async_llm = AsyncLLM.from_engine_args(
    AsyncEngineArgs(
        model="opendatalab/MinerU2.5-2509-1.2B",
        logits_processors=[MinerULogitsProcessor]  # if vllm>=0.10.1
    )
)

client = MinerUClient(
  backend="vllm-async-engine",
  vllm_async_llm=async_llm,
)

async def main():
    image_path = "/path/to/the/test/image.png"
    async with aiofiles.open(image_path, "rb") as f:
        image_data = await f.read()
    image = Image.open(io.BytesIO(image_data))
    extracted_blocks = await client.aio_two_step_extract(image)
    print(extracted_blocks)

asyncio.run(main())

async_llm.shutdown()
```

### `llama-cpp-engine` Example

This backend uses the in-process `mineru_llama_cpp.Engine` — no HTTP server needed.
The caller creates and manages the Engine lifecycle.

```python
from PIL import Image
from mineru_llama_cpp import Engine
from mineru_vl_utils import MinerUClient

with Engine("/path/to/model.gguf", "/path/to/mmproj.gguf") as engine:
    client = MinerUClient(backend="llama-cpp-engine", llama_cpp_engine=engine)
    image = Image.open("/path/to/the/test/image.png")
    extracted_blocks = client.two_step_extract(image)
    print(extracted_blocks)
```

## Other APIs

Besides the `two_step_extract` method, `MinerUClient` also provides other APIs
for interacting with the model. Following are the main APIs:

```python
class MinerUClient:

    def layout_detect(self, image: Image.Image) -> list[ContentBlock]:
        ...

    def batch_layout_detect(self, images: list[Image.Image]) -> list[list[ContentBlock]]:
        ...

    async def aio_layout_detect(self, image: Image.Image) -> list[ContentBlock]:
        ...

    async def aio_batch_layout_detect(self, images: list[Image.Image]) -> list[list[ContentBlock]]:
        ...

    def two_step_extract(self, image: Image.Image) -> list[ContentBlock]:
        ...

    def batch_two_step_extract(self, images: list[Image.Image]) -> list[list[ContentBlock]]:
        ...

    async def aio_two_step_extract(self, image: Image.Image) -> list[ContentBlock]:
        ...

    async def aio_batch_two_step_extract(self, images: list[Image.Image]) -> list[list[ContentBlock]]:
        ...
```

## Limitations

The `transformers` backend is slow and not suitable for production use.

The `MinerUClient` only supports standalone image(s) as input.
PDF and DOCX files are not planned to be supported.
Cross-page and cross-document operations are not planned to be supported, too.

For production use cases, please use [MinerU](https://github.com/opendatalab/mineru),
which is a more complete toolkit for document analyzing and data extraction.

### MLX model path lifecycle

`mineru_vl_utils.mlx_compat.prepare_mlx_model_path` (2.0.1+) resolves a local path or Hugging Face repo and prepares Qwen compatibility configuration without modifying source weights:

```python
from mineru_vl_utils.mlx_compat import prepare_mlx_model_path

with prepare_mlx_model_path("/path/to/model") as model_path:
    # Keep this context open for the entire server lifetime.
    run_server(model_path)
```

The context removes only its own temporary directory on exit, including failure paths. `load_mlx_model` uses the same preparation logic for local inference.

### MLX engine batches

MLX engine batches use the public mlx-vlm `BatchGenerator` while preserving the existing MinerU chat template and image position:

```python
client = MinerUClient("mlx-engine", model_path="/path/to/model", batch_size=8)
results = client.batch_content_extract(images)
```

`batch_size=0` (the default) selects 8; `batch_size=1` retains single-image generation. Inputs are grouped by effective sampling parameters and image/text modality, then sorted by pixel count. Results retain input order. Batches are limited by both the requested count and a 9,000,000-pixel input budget; an oversized image runs alone without resizing. This is a batching budget, not a hard memory limit: long contexts and outputs still consume KV cache memory.

Generation remains serialized with the same lock, including asynchronous calls; batching does not enable simultaneous GPU calls from multiple threads. Cancellation waits for in-flight work before releasing resources. Each sample gets independent penalty processors. As before, MLX does not implement `no_repeat_ngram_size` or priority scheduling. Batch decoding can produce small whitespace differences compared with sequential decoding.

The pixel budget accommodates eight 1036×1036 layout inputs (8,586,368 pixels). The default batch is 8; users can explicitly select `batch_size=4`, `2`, or `1` to reduce memory use. Increasing the budget does not enable concurrent GPU calls or remove per-batch limits.
