function directionTo(src, target) {
  const dx = target[0] - src[0];
  const dy = target[1] - src[1];
  if (dx === 0 && dy === 0) return 0;
  if (Math.abs(dx) > Math.abs(dy)) {
      return dx > 0 ? 2 : 4;
  } else {
      return dy > 0 ? 3 : 1;
  }
}

module.exports = { directionTo };