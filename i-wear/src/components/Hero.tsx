import InfoCards from "./InfoCards";

export default function Hero() {
  return (
    <div className="flex flex-col items-center drop-shadow-lg">
      <div className="bg-black rounded-t-lg border-black p-4 flex justify-center w-full">
        <h1 className="text-white text-4xl font-medium">One Press. Three Glasses.</h1>
      </div>
      <InfoCards />
    </div>
  );
}
