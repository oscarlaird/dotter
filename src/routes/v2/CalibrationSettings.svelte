<script>
    import Cookies from 'js-cookie';
    export let likelihood_model;
    function save_likelihood_model() {
        Cookies.set('likelihood_model', JSON.stringify(likelihood_model));
    }
</script>

<div class="flex flex-col border border-black rounded-md p-2">
    <div class="flex flex-row items-center gap-2">
        <label>Mean Delay:</label>
        <input type="range" min="-0.050" max="0.150" step="0.001" bind:value={likelihood_model.mu_delay} on:input={save_likelihood_model}/>
        <span>{likelihood_model.mu_delay.toFixed(3)}</span>
    </div>
    <div class="flex flex-row items-center gap-2">
        <label>Std Dev Delay:</label>
        <input type="range" min="0" max="0.150" step="0.001" bind:value={likelihood_model.stddev_delay} on:input={save_likelihood_model}/>
        <span>{likelihood_model.stddev_delay.toFixed(3)}</span>
    </div>
    <div class="flex flex-row items-center gap-2">
        <label>Outlier Prob:</label>
        <input type="range" min="0" max="0.150" step="0.001" bind:value={likelihood_model.outliers} on:input={save_likelihood_model}/>
        <span>{likelihood_model.outliers.toFixed(3)}</span>
    </div>
    <div class="flex flex-row items-center gap-2">
        <label>Period:</label>
        <input type="range" min="0.1" max="2.5" step="0.1" bind:value={likelihood_model.period} on:input={save_likelihood_model}/>
        <span>{likelihood_model.period.toFixed(1)}</span>
    </div>
</div>