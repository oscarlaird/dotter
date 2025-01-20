<script>
    // import Cookies from 'js-cookie';
    export let use_automatic_calibration;
    export let likelihood_model;
    export let auto_calibration_likelihood_model;
    $: if (use_automatic_calibration) {
        console.log("use_automatic_calibration", use_automatic_calibration);
        likelihood_model.mu_delay = auto_calibration_likelihood_model.mu_delay;
        likelihood_model.stddev_delay = auto_calibration_likelihood_model.stddev_delay;
        likelihood_model.outliers = auto_calibration_likelihood_model.outliers;
        likelihood_model.period = auto_calibration_likelihood_model.period;
    }
    function save_likelihood_model() {
        use_automatic_calibration = false;
        // Cookies.set('likelihood_model', JSON.stringify(likelihood_model));
    }
</script>
<div class="flex flex-col border border-gray-300 rounded-md bg-white p-1.5 gap-1.5">
    <div class="flex items-center gap-1.5">
        <label class="font-bold text-s">Auto Calibration</label>
        <input type="checkbox" bind:checked={use_automatic_calibration} />
    </div>

    <div class="grid grid-cols-2 gap-2">
        <div class="flex flex-col gap-0.5">
            <label class="font-bold text-s">Mean ({(1000 * likelihood_model.mu_delay).toFixed(0)}ms)</label>
            <input type="range" min="-0.050" max="0.150" step="0.001" bind:value={likelihood_model.mu_delay} on:input={save_likelihood_model} class="w-full"/>
        </div>

        <div class="flex flex-col gap-0.5">
            <label class="font-bold text-s">StdDev ({(1000 * likelihood_model.stddev_delay).toFixed(0)}ms)</label>
            <input type="range" min="0" max="0.150" step="0.001" bind:value={likelihood_model.stddev_delay} on:input={save_likelihood_model} class="w-full"/>
        </div>

        <div class="flex flex-col gap-0.5">
            <label class="font-bold text-s">Outliers ({(100 * likelihood_model.outliers).toFixed(1)}%)</label>
            <input type="range" min="0" max="0.150" step="0.001" bind:value={likelihood_model.outliers} on:input={save_likelihood_model} class="w-full"/>
        </div>

        <div class="flex flex-col gap-0.5">
            <label class="font-bold text-s">Period ({likelihood_model.period.toFixed(1)}s)</label>
            <input type="range" min="0.5" max="2.5" step="0.1" bind:value={likelihood_model.period} on:input={save_likelihood_model} class="w-full"/>
        </div>
    </div>
</div>