import { useRef } from "react";

export default function ImageSlots({ files, onChange, disabled }) {
  const input = useRef(null);
  const select = (index) => {
    if (disabled) return;
    input.current.dataset.index = index;
    input.current.click();
  };
  const receive = (event) => {
    const index = Number(event.currentTarget.dataset.index);
    const next = [...files];
    next[index] = event.target.files?.[0] || null;
    onChange(next);
    event.target.value = "";
  };
  const remove = (index) => onChange(files.map((file, i) => (i === index ? null : file)));

  return <>
    <input ref={input} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp"
      onChange={receive} />
    <div className="slots">
      {files.map((file, index) => <article className={`slot ${file ? "complete" : ""}`} key={index}>
        <button className="slot-image" type="button" onClick={() => select(index)} disabled={disabled}>
          {file ? <img src={URL.createObjectURL(file)} alt={`Selected view ${index + 1}`} /> : <span>＋<small>Photo {index + 1}</small></span>}
        </button>
        <div className="slot-meta"><span>{file ? file.name : "Add a view"}</span>
          {file && <button type="button" className="text-button" onClick={() => remove(index)} disabled={disabled}>Remove</button>}
        </div>
      </article>)}
    </div>
  </>;
}
