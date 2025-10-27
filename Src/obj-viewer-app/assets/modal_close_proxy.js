// Proxy clicks from in-modal Close buttons to the persistent hidden close trigger
document.addEventListener('click', function (e) {
    try {
        const target = e.target;
        if (!target) return;

        // Find the nearest button ancestor (handles clicks on spans/icons inside the button)
        const btn = target.closest && target.closest('button');
        if (!btn) return;

        // Only handle buttons whose visible text is exactly 'Close' (trimmed)
        const text = (btn.textContent || '').trim();
        if (text !== 'Close') return;

        // Ensure the clicked button is inside the modal container
        const modal = btn.closest && btn.closest('#global-descriptors-modal');
        if (!modal) return;

        const hidden = document.getElementById('global-descriptors-hidden-close-trigger');
        if (hidden) {
            // Prefer dispatching a MouseEvent in case some frameworks expect event properties
            try {
                hidden.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
            } catch (e) {
                // Fallback to `.click()` if dispatchEvent isn't supported
                hidden.click();
            }
        }
    } catch (err) {
        // Swallow errors to keep UI resilient
        console.debug('modal_close_proxy error:', err);
    }
});
