module.exports = ({core}) => {
  let force = core.getBooleanInput('force');
  let oldModels = JSON.parse(core.getInput('oldModels'));
  let newModels = JSON.parse(core.getInput('newModels'));
  let oldHash = core.getInput('oldHash');
  let newHash = core.getInput('newHash');

  if (oldHash !== newHash) {
    force = true;
  }
  if (force) {
    console.log("Workflow files or code changed; forcing full export");
    core.setOutput('to_export', JSON.stringify(newModels));
    core.setOutput('unchanged', JSON.stringify([]));
    return;
  }

  const oldMap = Object.fromEntries(oldModels.map((m) => [m.name, m]));
  const newMap = Object.fromEntries(newModels.map((m) => [m.name, m]));

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
