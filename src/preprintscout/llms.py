from openai import OpenAI


def llm(
    llm_prompt, temp, output_max, gpt_api_key=None, gemini_api_key=None, safety=None
):
    if gpt_api_key:
        client = OpenAI(api_key=gpt_api_key)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4-turbo-preview"]
            max_tokens=output_max,
            temperature=temp,
            messages=[
                {
                    "role": "system",
                    "content": "You are a research assistant, skilled in explaining complex research articles in plain language.",
                },
                {"role": "user", "content": llm_prompt},
            ],
        )

        return completion.choices[0].message
    elif gemini_api_key:

        # Palm is going to deprecating, update to Gemini by August
        import google.generativeai as palm

        palm.configure(api_key=gemini_api_key)
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
    else:
        print("No API keys provided for OpenAI or Google.")
