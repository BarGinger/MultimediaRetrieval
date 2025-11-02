// Proxy clicks from in-modal Close buttons to the persistent hidden close trigger
document.addEventListener('click', function (e) {
    try {
        const target = e.target;
        if (!target) return;

        // Find the nearest button ancestor (handles clicks on spans/icons inside the button)
        const btn = target.closest && target.closest('button');
        if (!btn) return;

        // Only handle buttons whose visible text is exactly 'Close' or '✕' (trimmed)
        const text = (btn.textContent || '').trim();
        if (text !== 'Close' && text !== '✕') return;

        // Ensure the clicked button is inside one of the modal containers
        const modalGlobal = btn.closest && btn.closest('#global-descriptors-modal');
        const modalAux = btn.closest && btn.closest('#aux-descriptors-modal');
        const modalClustering = btn.closest && btn.closest('#clustering-modal');
        if (!modalGlobal && !modalAux && !modalClustering) return;

        // Prefer the matching hidden trigger (priority: aux > clustering > global)
        let hidden = null;
        if (modalAux) {
            hidden = document.getElementById('aux-descriptors-hidden-close-trigger');
        } else if (modalClustering) {
            hidden = document.getElementById('clustering-modal-hidden-close-trigger');
        } else {
            hidden = document.getElementById('global-descriptors-hidden-close-trigger');
        }
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
