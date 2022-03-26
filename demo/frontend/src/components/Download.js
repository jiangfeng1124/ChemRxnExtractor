import React from "react";

export class Download extends React.Component {
	render() {
		return <div className="mt-5 mx-5">
			<h4> Downloads </h4>

			<p>
				ChemRxnExtractor is available as an open-source package. Check our <a href="https://github.com/jiangfeng1124/ChemRxnExtractor">GitHub Repository</a> for the code, data, API and usage instructions.
			</p>

			<p>
				Model Checkpoints <br />
				<a href='https://drive.google.com/file/d/1HeP2NlSAdqNzlTqmHCrwmoUNiw9JWdaf/view?usp=sharing'>cre_models_v01.tgz</a> (768MB)
			</p>
		</div>
	};
}