import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

export default function GaussianViewer({ modelUrl, onReset }) {
  const mount = useRef(null);
  const resetView = useRef(() => {});
  const [loading, setLoading] = useState(true);
  const [loadPercent, setLoadPercent] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!modelUrl || !mount.current) return undefined;
    setLoading(true); setError(""); setLoadPercent(null);
    const host = mount.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#102b36");
    const camera = new THREE.PerspectiveCamera(55, 1, 0.001, 10000);
    const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    let homePosition = new THREE.Vector3(0, 0, 3);
    let homeTarget = new THREE.Vector3();
    const applyHome = () => {
      camera.position.copy(homePosition);
      controls.target.copy(homeTarget);
      controls.update();
    };
    resetView.current = applyHome;
    applyHome();

    const spark = new SparkRenderer({ renderer });
    scene.add(spark);
    let splat;
    try {
      splat = new SplatMesh({
        url: modelUrl,
        onProgress: (event) => {
          if (event.lengthComputable && event.total > 0) setLoadPercent(Math.round(event.loaded / event.total * 100));
        },
        onLoad: (mesh) => {
          const sphere = mesh.getBoundingBox(true).getBoundingSphere(new THREE.Sphere());
          if (Number.isFinite(sphere.radius) && sphere.radius > 0) {
            homeTarget = sphere.center.clone();
            homePosition = sphere.center.clone().add(new THREE.Vector3(0, 0, Math.max(sphere.radius * 2.8, 0.1)));
            camera.near = Math.max(sphere.radius / 1000, 0.001);
            camera.far = Math.max(sphere.radius * 100, 100);
            camera.updateProjectionMatrix();
            applyHome();
          }
          setLoading(false);
        },
      });
      scene.add(splat);
      splat.initialized.catch((reason) => {
        setLoading(false);
        setError(reason instanceof Error ? reason.message : "The reconstruction model could not be loaded.");
      });
    } catch (reason) {
      setLoading(false);
      setError(reason instanceof Error ? reason.message : "The reconstruction model could not be loaded.");
    }

    const resize = () => {
      const { width, height } = host.getBoundingClientRect();
      if (!width || !height) return;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

    return () => {
      observer.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      splat?.dispose();
      spark.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      resetView.current = () => {};
    };
  }, [modelUrl]);

  const fullscreen = () => mount.current?.parentElement?.requestFullscreen?.();
  return <>
    <div className="viewer">
      <div className="viewer-canvas" ref={mount} />
      {loading && !error && <div className="viewer-overlay">Loading Gaussian Splat{loadPercent == null ? "…" : `… ${loadPercent}%`}</div>}
      {error && <div className="viewer-overlay viewer-error"><b>Unable to display this reconstruction.</b><span>{error}</span></div>}
    </div>
    <div className="viewer-actions"><span>Drag to rotate · Scroll or pinch to zoom · Right drag to pan</span><button onClick={() => resetView.current()}>Reset view</button><button onClick={fullscreen}>Fullscreen</button><button onClick={onReset}>Reconstruct another</button></div>
  </>;
}
