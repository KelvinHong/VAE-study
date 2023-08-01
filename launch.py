import gradio as gr
from gradio_utils import *

with gr.Blocks(title="Master: VAE Experiments", css=r"#dangerous {color: red}") as demo:
    gr.Markdown("# Variational Autoencoder Research - Web UI")
    with gr.Tab("Train Model"):
        with gr.Row():
            with gr.Column():
                with gr.Accordion(label="Training parameters"):
                    with gr.Row():
                        with gr.Column(scale=4):
                            model_name = gr.Dropdown(choices = find_existing_models(), 
                                                interactive=True, label="Select Model or create New Model",
                                                allow_custom_value=True) 
                        with gr.Column(min_width=30, scale=1):
                            resume_flag = gr.Checkbox(value=False, label="Resume from checkpoint?")
                        with gr.Column(min_width=30, scale=1):
                            refresh_train_model_button = gr.Button("Refresh Models")
                    model_type = gr.Dropdown(list(MODEL_TYPE_MAPPING.keys()), label="Model Type", value="AutoEncoder")
                    additional_hp = gr.Textbox(placeholder="Additional Information provided for training model", 
                                                interactive=True, label="Additional Hyperparameters (see example below)")
                    adhp_examples = gr.Examples(
                                examples=[
                                    r'{"beta": 5}', 
                                    r'{"beta": 4, "capacity": 50}', 
                                    r'{"alpha": 1, "lambda":1000}',
                                    r'{"beta": 4, "perceptual": 0.5}', 
                                    r'{"beta": 4, "capacity": 50, "perceptual": 0.3}', 
                                    r'{"alpha": 1, "lambda":1000, "perceptual": 0.5}',
                                    r'{"aggresive-steps": 3, "aggresive-cutoff-epoch": 5}',
                                ],
                                inputs=[additional_hp]
                            )
                    dataset = gr.Radio(list(COMPATIBLE_DATAMAP.keys()), label="Dataset")
                    batch_size = gr.Radio([8, 16, 32, 64, 128], value=32, label="Batch Size")
                    optimizer = gr.Radio(["Adam", "SGD"], label="Optimizer", value="Adam")
                    epochs = gr.Slider(0, 300, step=1, label="Epochs (Train until this epoch if enabled resume training)")
                    learning_rate = gr.Slider(minimum=0, maximum=0.2, label="Learning Rate")
                    with gr.Row():
                        with gr.Column(min_width=30, scale=1):
                            lr_scheduler_flag = gr.Checkbox(label="Use Learning Rate Scheduler", value=False)
                        with gr.Column(scale=4):
                            lr_scheduler = gr.Textbox(label="Input formula (see example)", visible=False)
                            examples = gr.Examples(examples=["ReduceLROnPlateau"],
                                            inputs=[lr_scheduler])
                    latent_dimension = gr.Slider(1, 300, step=1, label="Latent Dimension")
                    load_pref_bundle = [model_type, dataset, batch_size, optimizer, model_name, 
                                epochs, learning_rate, latent_dimension, 
                                lr_scheduler_flag, lr_scheduler, additional_hp]
                    input_bundle = load_pref_bundle + [resume_flag]
                training_bar = gr.Text(label="Training Result")
                with gr.Row():
                    with gr.Column(min_width=30):
                        if pref is not None:
                            pref_button = gr.Button("Load Preference")
                    with gr.Column(min_width=30):
                        headless_button = gr.Button("Submit to Headless (stackable)")
                    with gr.Column(min_width=30):
                        train_button = gr.Button("Begin Training")
                with gr.Row():
                    with gr.Column(scale=4):
                        headless_command = gr.Textbox(label="Run this in your terminal for headless training.",
                                                    interactive=False)
                    with gr.Column(min_width=30, scale=1):
                        headless_trigger = gr.Button("Train Remotely")
                    with gr.Column(min_width=30, scale=1):
                        headless_clear = gr.Button("Clear")
                
            with gr.Column():
                # images = [gr.Image(shape=(640,480), interactive=False) for _ in range(GRAPHS)]
                images = gr.Gallery(label="Metrics")
                show_metric_button = gr.Button("Show current metrics")

    with gr.Tab("Model Inferencing"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    trained_model_name = gr.Dropdown(choices = find_existing_models(), 
                                                interactive=True, label="Select Model",
                                                allow_custom_value=False)
                    refresh_model_button = gr.Button("Refresh Models", elem_id="refresh-model-button")
                with gr.Row():
                    compatible_datasets = gr.Dropdown(choices = [], interactive=True, 
                                        label="Select Dataset", allow_custom_value=False)
                    train_or_valid = gr.Radio(choices=["train", "valid"], value="valid",
                                                label="Choose training dataset or validation dataset")
                prompt_dataset = gr.Textbox(value="", label="Dataset Description")
                config_code = gr.Textbox(max_lines=10, 
                            label="Model Configuration",
                            placeholder="Model configurations will be shown here.")
            with gr.Column():
                with gr.Tab("Bulk Inferencing"):
                    with gr.Row():
                        seed_choser = gr.Number(value=-1, label="Choose a positive seed (-1 means not using seed)", interactive=True)
                        bulk_prompt = gr.Textbox(interactive=False)
                    bulk_inference_button = gr.Button("Bulk Inference")
                    bulk_input_images = gr.Image(interactive=False, label="Inputs")
                    bulk_output_images = gr.Image(interactive=False, label="Outputs")
                with gr.Tab("Prior Sampling"): 
                    sampling_button = gr.Button("Sampling")
                    sampling_output = gr.Image(interactive=False, label="Outputs")
                with gr.Tab("Latent Interpolation"):
                    with gr.Row():
                        inter_seed_choser = gr.Number(value=-1, label="Choose a positive seed (-1 means not using seed)", interactive=True)
                        inter_prompt = gr.Textbox(interactive=False)
                    inter_2 = gr.Button("Interpolate from two images")
                    inter_4 = gr.Button("Interpolate from four images")
                    interpolate_output = gr.Image(interactive=False)
                with gr.Tab("Latent Traversal"):
                    gr.Markdown("## Traverse latent from an image on every z-dim.")
                    with gr.Row():
                        lt_image_choser = gr.Slider(minimum=1, maximum=100, value=1, step=1,
                                            label=f"Choose image from [Dataset] [Train/Valid]",
                                            interactive=True)
                        lt_submit = gr.Button("Perform Latent Traversal")
                    lt_bound = gr.Number(value=2, label="Latent bound for ploting traversal, must be positive.")
                    lt_dimension = gr.Number(value=0, label="Latent Dimension", interactive=False)
                    lt_gallery = gr.Gallery(label="View latent traversal results here")
                with gr.Tab("Single Image Inferencing"):
                    with gr.Row():
                        image_choser = gr.Slider(minimum=1, maximum=100, value=1, step=1,
                                            label=f"Choose image from [Dataset] [Train/Valid]",
                                            interactive=True)
                        load_image_button = gr.Button("Load image")
                    with gr.Row():
                        input_image = gr.Image(interactive=False)
                        output_image = gr.Image(interactive=False)
                    inference_button = gr.Button("Inference")
    
    with gr.Tab("Inspect Models"):
        gr.Markdown("# View models performances here.")
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                last_nrows = gr.Slider(label="Show last n rows", maximum=50, step=1, value=10)
                view_result_button = gr.Button("View results")
            with gr.Column(scale=4):
                dataframes = [
                    gr.Dataframe(visible=False, interactive=False)
                        for _ in range(MAX_NUM_DATAFRAMES)
                ]

    with gr.Tab("T-SNE Analysis"):
        gr.Markdown("# Perform T-SNE to investigate the level of posterior collapse.")
        with gr.Row():
            with gr.Column(scale=2, min_width=100):
                with gr.Row():
                    with gr.Column(scale=4):
                        tsne_model_name = gr.Dropdown(choices = find_existing_models(), 
                                            interactive=True, label="Select Model") 
                    with gr.Column(min_width=30, scale=1):
                        tsne_refresh_model_button = gr.Button("Refresh Models")
                with gr.Row():
                    with gr.Column(scale=3, min_width=100):
                        tsne_dataset = gr.Dropdown(choices=[], interactive=True, label="Select dataset")
                    with gr.Column(scale=2, min_width=50):
                        tsne_split = gr.Dropdown(choices=["train", "valid"], value="valid", interactive=True, label="Select dataset portion")
                tsne_submit = gr.Button("Calculate t-SNE plot")
            with gr.Column(scale=5):
                tsne_output = gr.Image(interactive=False)

    with gr.Tab("Delete Model"):
        gr.Markdown("# Delete your models here, it is irreversible.")
        with gr.Row():
            existing_model_name = gr.Dropdown(choices = find_existing_models(), 
                                interactive=True, label="Select Model",
                                allow_custom_value=False)
            delete_button = gr.Button("Delete Model", elem_id="dangerous")
            delete_result = gr.Textbox(label="Status")

    if pref is not None:
        pref_button.click(
                        load_pref, 
                        outputs=load_pref_bundle, 
                    )
    lr_scheduler_flag.select(
                    toggle_lr_scheduler, 
                    inputs=lr_scheduler_flag, 
                    outputs=lr_scheduler
                )
    resume_flag.select(
                    load_from_model_name, 
                    inputs=[resume_flag, model_name], 
                    outputs=load_pref_bundle
                )
    headless_button.click(
                    headless_accept,
                    inputs = input_bundle,
                    outputs = headless_command,
                )
    headless_trigger.click(
                    headless_execute,
                    inputs = headless_command,
                )
    headless_clear.click(
                    headless_erase,
                    outputs = headless_command,
                )
    train_button.click(
                    accept_and_train, 
                    inputs = input_bundle,
                    outputs= training_bar
                )
    show_metric_button.click(
                    show_graphs, 
                    inputs=model_name, 
                    outputs=images
                )
    refresh_model_button.click(
                    find_existing_models_ui, 
                    outputs=trained_model_name
                )
    refresh_train_model_button.click(
                    find_existing_models_ui, 
                    outputs=model_name
                )
    data_select_inputs = [compatible_datasets, train_or_valid]
    data_select_outputs = [image_choser]
    data_select_outputs2 = [lt_image_choser]
    trained_model_name.select(
                    load_config_to_inference, 
                    inputs=trained_model_name, 
                    outputs=[compatible_datasets, prompt_dataset, config_code]
                ).then(
                    load_dataset, 
                    inputs=data_select_inputs, 
                    outputs=data_select_outputs2 # Load to latent traversal UI
                ).then(
                    load_dataset, 
                    inputs=data_select_inputs, 
                    outputs=data_select_outputs # Load to single inference UI
                ) # Load dataset immediately after loading model
    compatible_datasets.select(
                    load_dataset, 
                    inputs=data_select_inputs, 
                    outputs=data_select_outputs2 # Load to latent traversal UI
                ).then(
                    load_dataset, 
                    inputs=data_select_inputs, 
                    outputs=data_select_outputs # Load to single inference UI
                ) # Load dataset after selecting.
    delete_button.click(
                    delete_model, 
                    inputs=existing_model_name, 
                    outputs=delete_result
                )
    load_image_button.click(
                    load_dataset,
                    inputs=data_select_inputs, 
                    outputs=data_select_outputs
                ).then(
                    load_image_to_ui, 
                    inputs=image_choser, 
                    outputs = input_image
                )
    inference_button.click(
                    load_image_to_ui, 
                    inputs=image_choser, 
                    outputs = input_image
                ).then(
                    load_model, 
                    inputs=trained_model_name
                ).then(
                    inference, 
                    inputs=input_image,
                    outputs=output_image
                )
    bulk_inference_button.click(
                    load_model, 
                    inputs = trained_model_name
                ).then(
                    load_dataset,
                    inputs=data_select_inputs, 
                    outputs=data_select_outputs
                ).then(
                    verify_bulk_seed,
                    inputs = seed_choser,
                    outputs = bulk_prompt      
                ).then(
                    bulk_inference,
                    inputs = seed_choser,
                    outputs = [bulk_input_images, bulk_output_images]
                )
    view_result_button.click(
                    view_last_nrows, 
                    inputs=last_nrows,
                    outputs=dataframes
                )
    tsne_refresh_model_button.click(
                    find_existing_models_ui, 
                    outputs=tsne_model_name
                )
    tsne_model_name.select(
                    tsne_broadcast, 
                    inputs = tsne_model_name,
                    outputs = tsne_dataset
                )
    tsne_submit.click(
                    generate_tsne,
                    inputs = [tsne_model_name, tsne_dataset, tsne_split],
                    outputs = tsne_output
                )
    lt_submit.click(
                    load_model, 
                    inputs=trained_model_name
                ).then(
                    latent_traversal,
                    inputs = [lt_image_choser, lt_bound],
                    outputs = [lt_dimension, lt_gallery]
                )
    sampling_button.click(
                    load_model, 
                    inputs = trained_model_name            
                ).then(
                    sample_from_prior,
                    outputs = sampling_output
                )
    inter_2.click(
                    load_model, 
                    inputs = trained_model_name
                ).then(
                    load_dataset,
                    inputs=data_select_inputs, 
                    outputs=data_select_outputs
                ).then(
                    verify_bulk_seed,
                    inputs = inter_seed_choser,
                    outputs = inter_prompt,      
                ).then(
                    interpolate_two,
                    inputs = inter_seed_choser,
                    outputs = interpolate_output,
                )
    inter_4.click(
                    load_model, 
                    inputs = trained_model_name
                ).then(
                    load_dataset,
                    inputs=data_select_inputs, 
                    outputs=data_select_outputs
                ).then(
                    verify_bulk_seed,
                    inputs = inter_seed_choser,
                    outputs = inter_prompt,      
                ).then(
                    interpolate_four,
                    inputs = inter_seed_choser,
                    outputs = interpolate_output,
                )

demo.queue(concurrency_count=5, max_size=20)
demo.launch(
    server_port=4896, 
    max_threads=4, 
    share=False, 
)