module.exports = ({context, core, modelLists, hashes}) => {
  let force = false;
  if (context.eventName === 'workflow_dispatch' && core.getBooleanInput('force')) {
    force = true;
  }
  if (hashes.old !== hashes.new) {
    force = true;
  }
  if (force) {
    console.log("Workflow files or code changed; forcing full export");
    core.setOutput('to_export', JSON.stringify(modelLists.new));
    core.setOutput('unchanged', JSON.stringify([]));
    return;
  }

  const oldMap = Object.fromEntries(modelLists.old.map((m) => [m.name, m]));
  const newMap = Object.fromEntries(modelLists.new.map((m) => [m.name, m]));

  const keys = new Set([...Object.keys(oldMap), ...Object.keys(newMap)]);

  const to_export = [];
  const unchanged = [];

  function equal(m, n) {
    if (!m && !n) return true;
    if (!m || !n) return false;
    return m["name"] === n["name"] &&
      m["source"] === n["source"] &&
      m["hf-name"] === n["hf-name"];
  }

  for (const key of keys) {
    const o = oldMap[key];
    const n = newMap[key];
    const eq = equal(o, n);

    if (eq) {
      unchanged.push(n);
    }
    if (!eq && n) {
      to_export.push(n);
    }
    // !n: deleted, which we ignore
  }

  core.setOutput('to_export', JSON.stringify(to_export));
  core.setOutput('unchanged', JSON.stringify(unchanged));
}
