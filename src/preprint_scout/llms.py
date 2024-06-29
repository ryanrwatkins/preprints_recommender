import google.generativeai as palm

try:
    from .api_keys import hf_api_key, palm_api_key
except ImportError:
    from api_keys import hf_api_key, palm_api_key


def palm_llm(llm_prompt, temp, output_max, safety):
    palm.configure(api_key=palm_api_key)
    # this gets the latest model for text generation
    models = [
        m
        for m in palm.list_models()
        if "generateText" in m.supported_generation_methods
    ]
    model = models[0].name
    llm_completion = palm.generate_text(
        model=model,
        prompt=llm_prompt,
        temperature=temp,
        max_output_tokens=output_max,
        safety_settings=[
            {
                "category": 1,  # 1 to 6
                "threshold": safety,  # 1 is block most anything, 3 is high threshold, 4 is block none
            },
            {
                "category": 2,
                "threshold": safety,
            },
            {
                "category": 3,
                "threshold": safety,
            },
            {
                "category": 4,
                "threshold": safety,
            },
            {
                "category": 5,
                "threshold": safety,
            },
            {
                "category": 6,
                "threshold": safety,
            },
        ],
    )
    return llm_completion.result
