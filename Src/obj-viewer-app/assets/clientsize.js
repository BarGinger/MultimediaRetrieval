// Dash loads any JS in /assets automatically.
// This script highlights the selected file button whenever selected-file-store.data changes.

window.addEventListener('DOMContentLoaded', function () {
  // Subscribe to mutations on #_dash-app-content to catch Dash render
  const obs = new MutationObserver(() => {
    const store = document.querySelector('[data-dash-component="Store"][id="selected-file-store"]');
    if (!store) return;

    const reactProps = store.__dashprivate__ && store.__dashprivate__.props;
    if (!reactProps) return;

    // Monkey-patch setProps to intercept changes to "data"
    const origSetProps = store.setProps;
    store.setProps = function (newProps) {
      if (newProps && Object.prototype.hasOwnProperty.call(newProps, 'data')) {
        const idx = newProps.data;
        const allButtons = document.querySelectorAll('[data-file-index]');
        allButtons.forEach(btn => btn.classList.remove('file-button-selected'));
        const target = document.querySelector(`[data-file-index="${idx}"]`);
        if (target) target.classList.add('file-button-selected');
      }
      return origSetProps.apply(this, arguments);
    };
    obs.disconnect();
  });
  obs.observe(document.body, { childList: true, subtree: true });
});
