import ArchitectureDiagram from "./ArchitectureDiagram.jsx";

export default function About() {
  return (
    <section className="section narrow prose">
      <h1 className="h1-sm">About RetinaAI</h1>

      <h2>The problem</h2>
      <p>
        Diabetic retinopathy (DR) damages the small blood vessels of the retina in people with
        diabetes. Screening programmes photograph the back of the eye (a fundus image) and grade
        the severity. This project uses a convolutional neural network to sort fundus images into
        five levels: No DR, Mild, Moderate, Severe and Proliferative DR.
      </p>

      <h2>Model architecture</h2>
      <ArchitectureDiagram />
      <h3>CNN and transfer learning</h3>
      <p>
        The model uses an EfficientNetB0 convolutional backbone pretrained on ImageNet and adapted
        to retinal images. Its 7 × 7 × 1280 feature map is summarised by global average pooling,
        followed by batch normalization, dropout (0.3), a 256-unit ReLU dense layer, dropout (0.2)
        and a softmax layer that outputs five class probabilities.
      </p>

      <h2>Preprocessing</h2>
      <p>
        Every uploaded image is converted to RGB, resized to 224 × 224 and passed to the model as
        float32 pixel values. EfficientNetB0 performs its own input scaling and normalization
        internally.
      </p>

      <h2>Image quality check and Grad-CAM</h2>
      <p>
        Before classification, simple checks on resolution, brightness, contrast and sharpness
        reject clearly unusable images. Grad-CAM then uses the gradients flowing into the
        <code> top_activation</code> layer of the EfficientNetB0 backbone to draw a heatmap of the
        regions that most influenced the prediction. It shows the model's attention, not the
        location of lesions.
      </p>

      <h2>Limitations</h2>
      <ul>
        <li>The model has not been clinically validated and is not a certified medical device.</li>
        <li>Results may not transfer to other cameras, clinics or populations.</li>
        <li>Neighbouring grades (for example Mild and Moderate) are hard to separate.</li>
        <li>Grad-CAM shows model attention only; it is not a clinically validated lesion detector.</li>
        <li>The image quality check is a simple heuristic, not a clinical gradability test.</li>
      </ul>
      <p className="disclaimer">
        This application is intended for AI-assisted screening and educational/research purposes. It
        does not provide a medical diagnosis. Users should consult a qualified ophthalmologist for
        clinical evaluation.
      </p>
    </section>
  );
}
